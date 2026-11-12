"""Train MADDPG on mixed tasks with uniform replay."""

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
import Code.config.alg_cfgs.maddpg as algorithm_settings
import Code.config.train_maddpg as train_settings
from Code.src.buffer.maddpg_buffer import MADDPGReplayBuffer
from Code.src.env import Observation
from Code.src.env.sampling.baseline_env_pool import BaselineEnvPool
from Code.src.utils.metrics import BaselineEpisodeStats
from Code.src.utils.logger import BaselineTrainingLogger
from Code.src.utils.random import make_rng, set_seed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv=None):
    parse_args(argv)

    from Code.src.alg.baselines.maddpg import MADDPG


    train_cfg = parameters(train_settings)
    run_dirs = []
    for seed in train_cfg.seeds:
        print(f"Training seed={seed}", flush=True)
        env_cfg, sampling_cfg = parameters(env_settings), parameters(sampling_settings)
        reference_cfg, disturbance_cfg = parameters(reference_settings), parameters(disturbance_settings)
        maddpg_cfg = parameters(algorithm_settings)
        if train_cfg.update_every < 1 or train_cfg.gradient_steps < 1 or train_cfg.exploration_decay_steps < 1:
            raise ValueError("Update interval, gradient steps and exploration decay must be positive")
        if maddpg_cfg.batch_size < 1 or maddpg_cfg.batch_size > maddpg_cfg.replay_capacity:
            raise ValueError("Batch size must be positive and fit in replay capacity")

        set_seed(seed)
        rng = make_rng(seed, 5)
        dimensions = BaselineEnvPool.dimensions()
        params = dict(vars(maddpg_cfg), **dimensions)
        configs = dict(env=env_cfg, reference=reference_cfg, sampling=sampling_cfg,
                       disturbance=disturbance_cfg, maddpg=maddpg_cfg)
        with BaselineEnvPool(train_cfg.num_envs, env_cfg, sampling_cfg, reference_cfg,
                             disturbance_config=disturbance_cfg,
                             seed=seed, render=train_cfg.render) as pool:
            agent = MADDPG(params)
            replay = MADDPGReplayBuffer(params, pool.max_agents, seed=seed)
            logger = BaselineTrainingLogger(train_cfg, configs, seed=seed, algorithm=agent.algorithm, observation=dimensions)
            print(f"MADDPG run={logger.run_dir} envs={pool.num_envs} device={agent.device}", flush=True)
            start = perf_counter()
            try:
                global_step, episode = 0, 0
                current = pool.reset() if train_cfg.total_steps > 0 else []
                metadata = [r["episode"] for r in current]
                stats = [BaselineEpisodeStats() for _ in current]
                active = np.full(pool.num_envs, train_cfg.total_steps > 0, dtype=bool)
                while active.any():
                    before_step = global_step
                    env_ids = np.flatnonzero(active)
                    obs, state = pack_batch([current[i] for i in env_ids], pool.max_agents)
                    action = collect_action(agent, obs, train_cfg, global_step, rng)
                    results = pool.step(action, env_ids)
                    valid = np.array([r["info"]["valid_transition"] for r in results])
                    if valid.any():
                        accepted = [r for r, keep in zip(results, valid) if keep]
                        next_obs, next_state = pack_batch(accepted, pool.max_agents)
                        replay.add_batch(obs[valid], {key: value[valid] for key, value in state.items()},
                                          action[valid], [r["reward"] for r in accepted], next_obs, next_state,
                                          [r["terminated"] for r in accepted], [r["truncated"] for r in accepted])


                    draining = global_step + len(env_ids) >= train_cfg.total_steps
                    reset_ids = []
                    for env_id, result in zip(env_ids, results):
                        global_step += 1
                        stats[env_id].update(result["reward"], result["info"])
                        current[env_id] = result
                        if result["terminated"] or result["truncated"]:
                            episode += 1
                            row = logger.log_episode(stats[env_id], episode=episode,
                                                     global_step=global_step, env_id=env_id,
                                                     episode_metadata=metadata[env_id])
                            print(f"env={env_id} episode={episode} step={global_step} "
                                  f"env_return={row['env_return']:.3f} end={row['termination_reason']}", flush=True)
                            if draining:
                                active[env_id] = False
                            else:
                                reset_ids.append(env_id)
                    if reset_ids:
                        for env_id, initial in zip(reset_ids, pool.reset(reset_ids)):
                            current[env_id], metadata[env_id] = initial, initial["episode"]
                            stats[env_id].reset()

                    updates_due = global_step // train_cfg.update_every - before_step // train_cfg.update_every
                    if global_step >= train_cfg.warmup_steps and len(replay) >= maddpg_cfg.batch_size:
                        for _ in range(updates_due * train_cfg.gradient_steps):
                            batch = replay.sample(maddpg_cfg.batch_size)
                            losses = agent.learn(batch)
                            logger.log_learning(losses, agent.learn_steps)

                agent.save_model(logger.model_path)
                print(f"Finished: step={global_step} episode={episode} learn_step={agent.learn_steps} "
                      f"steps/s={global_step / (perf_counter() - start):.1f}\n"
                      f"Actor: {logger.model_path}", flush=True)
            finally:
                logger.close()
        run_dirs.append(logger.run_dir)
    return run_dirs


def pack_batch(results, max_agents):
    obs = np.zeros((len(results), max_agents, Observation.obs_dim), dtype=np.float32)
    agents = np.zeros((len(results), max_agents, Observation.local_dim), dtype=np.float32)
    mask = np.zeros((len(results), max_agents, 1), dtype=bool)
    for i, result in enumerate(results):
        n = len(result["obs"])
        obs[i, :n] = result["obs"]
        agents[i, :n] = result["state"]["agents"]
        mask[i, :n] = True
    return obs, {"agents": agents, "agent_mask": mask,
                 "global": np.stack([r["state"]["global"] for r in results])}


def exploration_std(config, global_step):
    fraction = np.clip((global_step - config.warmup_steps) / config.exploration_decay_steps, 0., 1.)
    return config.exploration_std_start + fraction * (config.exploration_std_end - config.exploration_std_start)


def collect_action(agent, obs, config, global_step, rng):
    if global_step < config.warmup_steps:
        return rng.uniform(-1, 1, (*obs.shape[:-1], Observation.action_dim)).astype(np.float32)
    action = agent.choose_action(obs)
    noise = rng.normal(0., exploration_std(config, global_step), action.shape)
    return np.clip(action + noise, -1, 1).astype(np.float32)


if __name__ == "__main__":
    main()
