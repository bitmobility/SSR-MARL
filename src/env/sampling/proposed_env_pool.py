"""Parallel task sampling with safety filtering and DOB updates."""

import multiprocessing as mp
import signal
import traceback

import numpy as np

from ..geometry.formations import FORMATIONS
from ..tasks.training_env import TrainingEnv
from .process_io import receive_result, send_requests, close_workers
from .training_schedule import BalancedCaseSampler
from ...cbf import SafetyFilter


_INFO_FIELDS = (
    "valid_transition", "path_type", "elapsed_time", "mean_squared_attachment_error",
    "max_attachment_error", "path_error", "heading_error", "success", "collision_pairs",
    "coordination_violation", "path_deviation", "no_forward_reference", "numerical_failure",
    "termination_reason",
)


def _critic_state(env, safety, obs):
    return env.observation.critic_state(
        obs, env.vehicles, env.payload, safety.observer.get_state())


def _step(env, safety, action, use_safety_layer=True):
    state = env.get_state()
    nominal = env.scale_action(action)
    if use_safety_layer:
        filtered, cbf = safety.filter(state, nominal)
    else:
        filtered = nominal
        cbf = dict(correction_cost=0.0, fallback_used=False, status="disabled")
    obs, reward, terminated, truncated, info = env.step(filtered, policy_action=action)
    critic = None
    if info["valid_transition"]:
        safety.observer.update(state, info["executed_action"], info["full_state"], info["elapsed_time"])
        critic = _critic_state(env, safety, obs)
    if env.render_mode is not None:
        env.render(phase="Proposed", info=info)
        if env.render_closed:
            raise KeyboardInterrupt("Viewer closed")
    return dict(obs=obs, state=critic, reward=reward, terminated=terminated, truncated=truncated,
                info={key: info[key] for key in _INFO_FIELDS},
                cbf={key: cbf[key] for key in ("correction_cost", "fallback_used", "status")})


def _worker(connection, env_config, sampling, reference, safety_config, disturbance_config,
            seed, env_id, render, use_safety_layer):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env = None
    try:
        env = TrainingEnv(env_config, sampling=sampling, reference=reference,
                                disturbance_config=disturbance_config, seed=seed, env_id=env_id,
                                render_mode="human" if render else None)

        safety = SafetyFilter(env, safety_config)
        connection.send(("ok", None))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                obs, _ = env.reset(options=payload)
                safety.reset(env)
                result = dict(obs=obs, state=_critic_state(env, safety, obs),
                              episode=env.episode_metadata())
            elif command == "step":
                result = _step(env, safety, payload, use_safety_layer)
            else:
                raise ValueError(f"Unknown environment command: {command}")
            connection.send(("ok", result))
    except EOFError:
        pass
    except (Exception, KeyboardInterrupt) as error:
        kind = "interrupt" if isinstance(error, KeyboardInterrupt) else "error"
        try:
            connection.send((kind, traceback.format_exc()))
        except OSError:
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class ProposedEnvPool:
    def __init__(self, num_envs, env_config, sampling, reference, safety_config,
                 *, disturbance_config=None, seed=2026, render=False,
                 use_safety_layer=True):
        if num_envs < 1:
            raise ValueError("Environment count must be positive")
        self.num_envs = num_envs
        self.sampling = sampling
        self.case_sampler = BalancedCaseSampler(sampling, seed)
        self.max_agents = max(len(FORMATIONS[f][0]) for f in sampling.formations)
        self.n_agents = np.zeros(num_envs, dtype=int)
        self.connections, self.processes = [], []
        self.closed = False
        context = mp.get_context("spawn")
        try:
            for env_id in range(num_envs):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker,
                    args=(child, env_config, sampling, reference, safety_config, disturbance_config,
                          seed + env_id, env_id, render and env_id == 0, use_safety_layer),
                    name=f"transport-{env_id}")
                self.connections.append(parent)
                try:
                    process.start()
                    self.processes.append(process)
                finally:
                    child.close()
            for env_id in range(num_envs):
                self._receive(env_id)
        except BaseException:
            self.close()
            raise

    def _receive(self, env_id):
        return receive_result(self.connections[env_id], self.processes[env_id], env_id)

    def _request(self, command, payloads, env_ids):
        env_ids = list(env_ids)
        send_requests(self.connections, command, payloads, env_ids)
        return [self._receive(env_id) for env_id in env_ids]

    def reset(self, env_ids=None):
        env_ids = list(range(self.num_envs) if env_ids is None else env_ids)
        options = [self.case_sampler.next_options() for _ in env_ids]
        results = self._request("reset", options, env_ids)
        for env_id, result in zip(env_ids, results):
            self.n_agents[env_id] = result["episode"]["n_agents"]
        return results

    def step(self, actions, env_ids):
        env_ids = list(env_ids)

        commands = [action[:self.n_agents[i]] for action, i in zip(actions, env_ids)]
        return self._request("step", commands, env_ids)

    def close(self):
        if self.closed:
            return
        self.closed = True
        close_workers(self.connections, self.processes)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
