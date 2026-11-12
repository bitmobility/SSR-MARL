"""Train recurrent IPPO on mixed tasks."""

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
import Code.config.alg_cfgs.ippo as algorithm_settings
import Code.config.train_ippo as train_settings
from Code.src.buffer.ippo_buffer import EpisodeTrajectory, IPPORolloutBuffer
from Code.src.env import Observation
from Code.src.env.sampling.baseline_env_pool import BaselineEnvPool
from Code.src.env.tasks.ippo_env import IPPOEnv
from Code.src.utils.metrics import BaselineEpisodeStats
from Code.src.utils.logger import BaselineTrainingLogger
from Code.src.utils.random import set_seed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv=None):
    parse_args(argv)
    from Code.src.alg.baselines.ippo import IPPO


    train_cfg = parameters(train_settings)
    run_dirs = []
    for seed in train_cfg.seeds:
        print(f"Training seed={seed}", flush=True)
        env_cfg, sampling_cfg = parameters(env_settings), parameters(sampling_settings)
        reference_cfg, disturbance_cfg = parameters(reference_settings), parameters(disturbance_settings)
        ippo_cfg = parameters(algorithm_settings)
        if train_cfg.batch_steps < 1:
            raise ValueError("Batch steps must be positive")

        set_seed(seed)
        dimensions = dict(obs_dim=Observation.obs_dim, action_dim=Observation.action_dim)
        params = dict(vars(ippo_cfg), **dimensions)
        configs = dict(env=env_cfg, sampling=sampling_cfg, reference=reference_cfg,
                       disturbance=disturbance_cfg, ippo=ippo_cfg)
        with BaselineEnvPool(train_cfg.num_envs, env_cfg, sampling_cfg, reference_cfg,
                             disturbance_config=disturbance_cfg, seed=seed,
                             render=train_cfg.render, environment_class=IPPOEnv) as pool:
            agent, rollout = IPPO(params), IPPORolloutBuffer(params)
            logger = BaselineTrainingLogger(train_cfg, configs, seed=seed, algorithm=agent.algorithm,
                                             observation=dimensions)
            print(f"IPPO run={logger.run_dir} envs={pool.num_envs} device={agent.device}", flush=True)
            start = perf_counter()
            try:
                global_step, episode = 0, 0
                while global_step < train_cfg.total_steps:
                    batch_start = global_step
                    current = pool.reset() if train_cfg.total_steps > 0 else []
                    metadata = [r["episode"] for r in current]
                    stats = [BaselineEpisodeStats() for _ in current]
                    trajectories = [EpisodeTrajectory() for _ in current]
                    actor_memory = [agent.initial_state(len(r["obs"])) for r in current]
                    critic_memory = [agent.initial_state(len(r["obs"])) for r in current]
                    active = np.full(pool.num_envs, train_cfg.total_steps > 0, dtype=bool)

                    while active.any():
                        env_ids = np.flatnonzero(active)
                        observations = [current[i]["obs"] for i in env_ids]
                        sizes = [len(obs) for obs in observations]
                        splits = np.cumsum(sizes)[:-1]
                        action, raw, log_prob, value, next_actor, next_critic = agent.act(
                            np.concatenate(observations), join_memory([actor_memory[i] for i in env_ids]),
                            join_memory([critic_memory[i] for i in env_ids]))
                        actions, raws, log_probs, values = [np.split(x, splits) for x in (action, raw, log_prob, value)]
                        for j, (am, cm) in enumerate(zip(split_memory(next_actor, sizes), split_memory(next_critic, sizes))):
                            actor_memory[env_ids[j]], critic_memory[env_ids[j]] = am, cm

                        results = pool.step(actions, env_ids)
                        next_values = [np.zeros(n, dtype=np.float32) for n in sizes]
                        bootstrap = [j for j, r in enumerate(results) if r["info"]["valid_transition"] and not r["terminated"]]
                        if bootstrap:
                            next_obs = np.concatenate([results[j]["obs"] for j in bootstrap])
                            next_state = join_memory([critic_memory[env_ids[j]] for j in bootstrap])
                            predicted = agent.value(next_obs, next_state)
                            for j, v in zip(bootstrap, np.split(predicted, np.cumsum([sizes[j] for j in bootstrap])[:-1])):
                                next_values[j] = v


                        round_end = global_step + len(env_ids)
                        draining = (round_end - batch_start >= train_cfg.batch_steps
                                    or round_end >= train_cfg.total_steps)
                        reset_ids = []
                        for j, env_id in enumerate(env_ids):
                            result = results[j]
                            terminated, truncated = result["terminated"], result["truncated"]
                            if result["info"]["valid_transition"]:
                                trajectories[env_id].add(
                                    observations[j], actions[j], raws[j], log_probs[j],
                                    result["reward"], values[j], next_values[j], terminated, truncated)
                            global_step += 1
                            stats[env_id].update(np.mean(result["reward"]), result["info"])
                            current[env_id] = result
                            if terminated or truncated:
                                if len(trajectories[env_id]):
                                    rollout.add_episode(trajectories[env_id])
                                episode += 1
                                row = logger.log_episode(stats[env_id], episode=episode,
                                                         global_step=global_step, env_id=env_id,
                                                         episode_metadata=metadata[env_id])
                                print(f"env={env_id} episode={episode} step={global_step} "
                                      f"env_return={row['env_return']:.3f} "
                                      f"end={row['termination_reason']}", flush=True)
                                if draining:
                                    active[env_id] = False
                                else:
                                    reset_ids.append(env_id)
                        if reset_ids:
                            reset = pool.reset(reset_ids)
                            for env_id, initial in zip(reset_ids, reset):
                                current[env_id], metadata[env_id] = initial, initial["episode"]
                                stats[env_id].reset()
                                trajectories[env_id] = EpisodeTrajectory()
                                actor_memory[env_id] = agent.initial_state(len(initial["obs"]))
                                critic_memory[env_id] = agent.initial_state(len(initial["obs"]))

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


def join_memory(states):
    import torch

    return tuple(torch.cat(parts, dim=1) for parts in zip(*states))


def split_memory(state, sizes):
    return list(zip(*(part.split(sizes, dim=1) for part in state)))


if __name__ == "__main__":
    main()
