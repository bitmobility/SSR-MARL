"""Train Ours with mixed tasks and DOB-CBF feedback."""

from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
import argparse
import os
from pathlib import Path
import sys
from time import perf_counter


for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

if __package__ in (None, ""):
    spec = ModuleSpec("Code", loader=None, is_package=True)
    spec.submodule_search_locations = [str(Path(__file__).resolve().parents[1])]
    sys.modules["Code"] = module_from_spec(spec)

import numpy as np

from Code.config import parameters
import Code.config.env_cfgs.transport as env_settings
import Code.config.env_cfgs.sampling as sampling_settings
import Code.config.env_cfgs.disturbance as disturbance_settings
import Code.config.env_cfgs.reference as reference_settings
import Code.config.env_cfgs.safety as safety_settings
import Code.config.alg_cfgs.proposed as algorithm_settings
import Code.config.train_proposed as train_settings
from Code.src.buffer.proposed_buffer import ProposedRolloutBuffer
from Code.src.env import Observation
from Code.src.env.sampling.proposed_env_pool import ProposedEnvPool
from Code.src.utils.metrics import EpisodeStats
from Code.src.utils.logger import ProposedTrainingLogger
from Code.src.utils.random import set_seed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv=None):
    parse_args(argv)
    return run_training(parameters(train_settings))


def run_training(train_cfg, *, safety_config=None, ablation=None):
    from Code.src.alg.proposed import Proposed


    use_safety_layer = ablation.use_safety_layer if ablation is not None else True
    run_dirs = []
    for seed in train_cfg.seeds:
        print(f"Training seed={seed}", flush=True)
        env_cfg = parameters(env_settings)
        sampling_cfg = parameters(sampling_settings)
        disturbance_cfg = parameters(disturbance_settings)
        reference_cfg = parameters(reference_settings)
        proposed_cfg = parameters(algorithm_settings)
        safety_cfg = parameters(safety_settings if safety_config is None else safety_config)

        set_seed(seed)
        params = dict(vars(proposed_cfg), **Observation.dimensions())
        configs = dict(env=env_cfg, reference=reference_cfg, sampling=sampling_cfg,
                       proposed=proposed_cfg, safety=safety_cfg, disturbance=disturbance_cfg)
        if ablation is not None:
            configs["ablation"] = ablation

        with ProposedEnvPool(train_cfg.num_envs, env_cfg, sampling_cfg, reference_cfg, safety_cfg,
                          disturbance_config=disturbance_cfg,
                          seed=seed, render=train_cfg.render,
                          use_safety_layer=use_safety_layer) as pool:
            agent = Proposed(params)
            rollout = ProposedRolloutBuffer(params, pool.max_agents, train_cfg.rollout_steps, pool.num_envs)
            logger = ProposedTrainingLogger(train_cfg, configs, seed=seed)
            print(f"Proposed run={logger.run_dir} envs={pool.num_envs} device={agent.device}", flush=True)
            start = perf_counter()
            try:
                global_step, episode = 0, 0
                current = pool.reset() if train_cfg.total_steps > 0 else []
                metadata = [r["episode"] for r in current]
                stats = [EpisodeStats() for _ in range(pool.num_envs)]
                active = np.full(pool.num_envs, train_cfg.total_steps > 0, dtype=bool)
                while active.any():
                    env_ids = np.flatnonzero(active)
                    obs, state = pack_batch([current[i] for i in env_ids], pool.max_agents)
                    action, raw, log_prob, value = agent.act(obs, state)
                    results = pool.step(action, env_ids)
                    terminated = np.array([r["terminated"] for r in results])
                    truncated = np.array([r["truncated"] for r in results])
                    valid = np.array([r["info"]["valid_transition"] for r in results])
                    cbf_reward = train_cfg.cbf_reward_weight * np.expm1(
                        -np.array([r["cbf"]["correction_cost"] for r in results]) / train_cfg.cbf_reward_sigma**2)
                    reward = np.array([r["reward"] for r in results]) + cbf_reward


                    next_value = np.zeros((len(env_ids), 1), dtype=np.float32)
                    bootstrap = np.flatnonzero(valid & ~terminated)
                    if len(bootstrap):
                        _, next_state = pack_batch([results[j] for j in bootstrap], pool.max_agents)
                        next_value[bootstrap] = agent.value(next_state)
                    rollout.add_batch(env_ids, obs, state, action, raw, log_prob, reward, value, next_value,
                                      terminated, truncated, valid)


                    draining = global_step + len(env_ids) >= train_cfg.total_steps
                    reset_ids = []
                    for j, env_id in enumerate(env_ids):
                        result = results[j]
                        global_step += 1
                        stats[env_id].update(result["reward"], cbf_reward[j], result["info"], result["cbf"])
                        current[env_id] = result
                        if terminated[j] or truncated[j]:
                            episode += 1
                            row = logger.log_episode(
                                stats[env_id], episode=episode, global_step=global_step,
                                env_id=env_id, episode_metadata=metadata[env_id])
                            print(f"env={env_id} episode={episode} step={global_step} "
                                  f"env_return={row['env_return']:.3f} cbf_return={row['cbf_return']:.3f} "
                                  f"end={row['termination_reason']}", flush=True)
                            if draining:
                                active[env_id] = False
                            else:
                                reset_ids.append(env_id)
                    if reset_ids:
                        for env_id, initial in zip(reset_ids, pool.reset(reset_ids)):
                            current[env_id] = initial
                            metadata[env_id] = initial["episode"]
                            stats[env_id].reset()
                    if rollout.full:
                        if len(rollout):
                            rollout.compute_returns()
                            for learn_step, losses in agent.learn(rollout):
                                logger.log_learning(losses, learn_step)
                        rollout.clear()
                if len(rollout):
                    rollout.compute_returns()
                    for learn_step, losses in agent.learn(rollout):
                        logger.log_learning(losses, learn_step)
                rollout.clear()

                agent.save_model(logger.model_path)
                print(f"Finished: step={global_step} episode={episode} learn_step={agent.learn_steps} "
                      f"steps/s={global_step / (perf_counter() - start):.1f}\n"
                      f"Actor: {logger.model_path}", flush=True)
            finally:
                logger.close()
        run_dirs.append(logger.run_dir)
    return run_dirs


def pack_batch(results, max_agents):
    size = len(results)
    obs = np.zeros((size, max_agents, Observation.obs_dim), dtype=np.float32)
    agents = np.zeros((size, max_agents, Observation.agent_state_dim), dtype=np.float32)
    mask = np.zeros((size, max_agents, 1), dtype=bool)
    for i, result in enumerate(results):
        n = len(result["obs"])
        obs[i, :n] = result["obs"]
        agents[i, :n] = result["state"]["agents"]
        mask[i, :n] = True
    state = {"agents": agents, "agent_mask": mask,
             "global": np.stack([r["state"]["global"] for r in results])}
    return obs, state


if __name__ == "__main__":
    main()
