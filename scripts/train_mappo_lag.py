"""Train MAPPO-Lag on mixed tasks."""

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
import Code.config.alg_cfgs.mappo_lag as algorithm_settings
import Code.config.train_mappo_lag as train_settings
from Code.src.buffer.mappo_lag_buffer import MAPPOLagRolloutBuffer
from Code.src.env import Observation
from Code.src.env.sampling.baseline_env_pool import BaselineEnvPool
from Code.src.utils.metrics import MAPPOLagEpisodeStats
from Code.src.utils.logger import MAPPOLagTrainingLogger
from Code.src.utils.random import set_seed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv=None):
    parse_args(argv)

    from Code.src.alg.baselines.mappo_lag import MAPPOLag


    train_cfg = parameters(train_settings)
    run_dirs = []
    for seed in train_cfg.seeds:
        print(f"Training seed={seed}", flush=True)
        env_cfg, sampling_cfg = parameters(env_settings), parameters(sampling_settings)
        reference_cfg, disturbance_cfg = parameters(reference_settings), parameters(disturbance_settings)
        lag_cfg = parameters(algorithm_settings)

        set_seed(seed)
        dimensions = BaselineEnvPool.dimensions()
        params = dict(vars(lag_cfg), **dimensions)
        configs = dict(env=env_cfg, reference=reference_cfg, sampling=sampling_cfg,
                       disturbance=disturbance_cfg, mappo_lag=lag_cfg)
        with BaselineEnvPool(train_cfg.num_envs, env_cfg, sampling_cfg, reference_cfg,
                             disturbance_config=disturbance_cfg,
                             seed=seed, render=train_cfg.render) as pool:
            agent = MAPPOLag(params)
            rollout = MAPPOLagRolloutBuffer(params, pool.max_agents, train_cfg.rollout_steps, pool.num_envs)
            logger = MAPPOLagTrainingLogger(train_cfg, configs, seed=seed, algorithm=agent.algorithm, observation=dimensions)
            print(f"MAPPO-Lag run={logger.run_dir} envs={pool.num_envs} device={agent.device}", flush=True)
            start = perf_counter()
            try:
                global_step, episode = 0, 0
                current = pool.reset() if train_cfg.total_steps > 0 else []
                metadata = [r["episode"] for r in current]
                stats = [MAPPOLagEpisodeStats() for _ in current]
                completed_costs = []
                active = np.full(pool.num_envs, train_cfg.total_steps > 0, dtype=bool)
                while active.any():
                    env_ids = np.flatnonzero(active)
                    obs, state = pack_batch([current[i] for i in env_ids], pool.max_agents)
                    action, raw, log_prob, value, cost_value = agent.act(obs, state)
                    results = pool.step(action, env_ids)
                    terminated = np.array([r["terminated"] for r in results])
                    truncated = np.array([r["truncated"] for r in results])
                    valid = np.array([r["info"]["valid_transition"] for r in results])
                    reward = np.array([r["reward"] for r in results])
                    cost = np.array([safety_cost(r["info"]) for r in results])


                    next_value = np.zeros((len(env_ids), 1), dtype=np.float32)
                    next_cost_value = np.zeros_like(next_value)
                    bootstrap = np.flatnonzero(valid & ~terminated)
                    if len(bootstrap):
                        _, next_state = pack_batch([results[j] for j in bootstrap], pool.max_agents)
                        predicted_reward, predicted_cost = agent.value(next_state)
                        next_value[bootstrap] = predicted_reward
                        next_cost_value[bootstrap] = predicted_cost * ~truncated[bootstrap, None]
                    rollout.add_batch(env_ids, obs, state, action, raw, log_prob, reward, cost,
                                      value, cost_value, next_value, next_cost_value, terminated, truncated, valid)

                    draining = global_step + len(env_ids) >= train_cfg.total_steps
                    reset_ids = []
                    for j, env_id in enumerate(env_ids):
                        result = results[j]
                        global_step += 1
                        stats[env_id].update(result["reward"], result["info"], cost[j])
                        current[env_id] = result
                        if terminated[j] or truncated[j]:
                            episode += 1
                            row = logger.log_episode(stats[env_id], episode=episode,
                                                     global_step=global_step, env_id=env_id,
                                                     episode_metadata=metadata[env_id])

                            if not stats[env_id].numerical_failure:
                                completed_costs.append(stats[env_id].episode_cost)
                            print(f"env={env_id} episode={episode} step={global_step} "
                                  f"env_return={row['env_return']:.3f} cost={row['episode_cost']:.0f} "
                                  f"end={row['termination_reason']}", flush=True)
                            if draining:
                                active[env_id] = False
                            else:
                                reset_ids.append(env_id)
                    if reset_ids:
                        for env_id, initial in zip(reset_ids, pool.reset(reset_ids)):
                            current[env_id], metadata[env_id] = initial, initial["episode"]
                            stats[env_id].reset()
                    if rollout.full:
                        constraint_metrics = agent.update_lagrange(completed_costs)
                        completed_costs.clear()
                        if len(rollout):
                            rollout.compute_returns()
                            for index, (learn_step, losses) in enumerate(agent.learn(rollout)):
                                if index == 0:
                                    losses.update(constraint_metrics)
                                logger.log_learning(losses, learn_step)
                        rollout.clear()
                constraint_metrics = agent.update_lagrange(completed_costs)
                completed_costs.clear()
                if len(rollout):
                    rollout.compute_returns()
                    for index, (learn_step, losses) in enumerate(agent.learn(rollout)):
                        if index == 0:
                            losses.update(constraint_metrics)
                        logger.log_learning(losses, learn_step)
                rollout.clear()

                agent.save_model(logger.model_path)
                print(f"Finished: step={global_step} episode={episode} learn_step={agent.learn_steps} "
                      f"lagrange={agent.lagrange:.3f} steps/s={global_step / (perf_counter() - start):.1f}\n"
                      f"Actor: {logger.model_path}", flush=True)
            finally:
                logger.close()
        run_dirs.append(logger.run_dir)
    return run_dirs


def safety_cost(info):
    return float(info["valid_transition"]
                 and (info["coordination_violation"] or bool(info["collision_pairs"])))


def pack_batch(results, max_agents):
    size = len(results)
    obs = np.zeros((size, max_agents, Observation.obs_dim), dtype=np.float32)
    agents = np.zeros((size, max_agents, Observation.local_dim), dtype=np.float32)
    mask = np.zeros((size, max_agents, 1), dtype=bool)
    for i, result in enumerate(results):
        n = len(result["obs"])
        obs[i, :n] = result["obs"]
        agents[i, :n] = result["state"]["agents"]
        mask[i, :n] = True
    return obs, {"agents": agents, "agent_mask": mask,
                 "global": np.stack([r["state"]["global"] for r in results])}


if __name__ == "__main__":
    main()
