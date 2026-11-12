"""Training snapshots, episode logs and optional SwanLab metrics."""

from collections import deque
import csv

from datetime import datetime
import json
from pathlib import Path

import numpy as np

import yaml


def create_experiment_dir(output_dir, experiment_name, seed, label=None):
    prefix = f"{label}_" if label else ""
    run_name = f"{prefix}seed{seed}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    directory = Path(output_dir) / experiment_name / run_name
    (directory / "logs").mkdir(parents=True)
    return directory


def _open_episode_csv(run_dir, fields):
    file = (run_dir / "logs" / "episodes.csv").open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    file.flush()
    return file, writer


def _start_swanlab(train_config, run_dir, snapshot):
    if train_config.swanlab_enabled:
        import swanlab

        return swanlab.init(
            project=train_config.swanlab_project,
            experiment_name=f"{train_config.experiment_name}_{run_dir.name}",
            config=snapshot, logdir=str(run_dir / "logs" / "swanlab"))
    return None


def _write_training_episode(stats, episode_metadata, *, identity, csv_file, csv_writer,
                            recent_successes, swanlab_run):
    metadata = dict(episode_metadata)
    metadata["disturbance_parameters"] = json.dumps(
        np.asarray(metadata["disturbance_parameters"]).tolist(), separators=(",", ":"))
    row = dict(**identity, **metadata, **stats.result())
    csv_writer.writerow(row)
    csv_file.flush()
    recent_successes.append(row["success"])
    if swanlab_run is not None:
        swanlab_run.log({
            "train/episode_return": row["env_return"],
            "train/success_rate": float(np.mean(recent_successes)),
            "train/attachment_mean": row["attachment_mean"],
            "train/path_rmse": row["path_rmse"],
        }, step=int(row["global_step"]))
    return row


class ProposedTrainingLogger:
    episode_fields = (
        "seed", "episode", "global_step", "path_type", "episode_steps",
        "env_return", "cbf_return", "attachment_mean", "attachment_max",
        "path_rmse", "path_max", "heading_rmse", "success", "collision",
        "coordination_violation", "path_deviation", "no_forward_reference", "tracking_failure",
        "numerical_failure", "duration", "termination_reason",
        "qp_failures", "qp_failure_statuses",
        "env_id", "formation", "n_agents", "episode_seed",
        "disturbance_scale", "disturbance_parameters",
        "initial_lateral_offset", "initial_heading_offset",
    )

    def __init__(self, train_config, configs, *, seed):
        self.seed = seed
        self.recent_successes = deque(maxlen=train_config.outcome_window)
        self.run_dir = create_experiment_dir(train_config.output_dir, train_config.experiment_name, self.seed,
                                             label="mixed")
        self.model_path = self.run_dir / "checkpoints" / "actor.pt"
        self.model_path.parent.mkdir()
        snapshot = {name: vars(config) for name, config in dict(configs, train=train_config).items()}
        snapshot["train"]["seed"] = seed
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8")

        self.swanlab_run = _start_swanlab(train_config, self.run_dir, snapshot)

        self.csv_file, self.csv_writer = _open_episode_csv(self.run_dir, self.episode_fields)


    def log_episode(self, stats, *, episode, global_step, env_id, episode_metadata):
        return _write_training_episode(
            stats, episode_metadata,
            identity=dict(seed=self.seed, episode=episode,
                          global_step=global_step, env_id=int(env_id)),
            csv_file=self.csv_file, csv_writer=self.csv_writer,
            recent_successes=self.recent_successes, swanlab_run=self.swanlab_run)

    def log_learning(self, losses, learn_step):
        if self.swanlab_run is not None:
            self.swanlab_run.log({
                f"learn/{key}": float(losses[key])
                for key in ("actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction")
            }, step=int(learn_step))

    def close(self):
        self.csv_file.close()
        if self.swanlab_run is not None:
            self.swanlab_run.finish()


class BaselineTrainingLogger:
    episode_fields = (
        "seed", "episode", "global_step", "path_type", "episode_steps",
        "env_return", "attachment_mean", "attachment_max",
        "path_rmse", "path_max", "heading_rmse", "success", "collision",
        "coordination_violation", "path_deviation", "no_forward_reference", "tracking_failure",
        "numerical_failure", "duration", "termination_reason",
        "env_id", "formation", "n_agents", "episode_seed",
        "disturbance_scale", "disturbance_parameters",
        "initial_lateral_offset", "initial_heading_offset",
    )

    def __init__(self, train_config, configs, *, seed, algorithm, observation):
        self.seed = seed
        self.recent_successes = deque(maxlen=train_config.outcome_window)
        self.run_dir = create_experiment_dir(train_config.output_dir, train_config.experiment_name, self.seed,
                                             label="mixed")
        self.model_path = self.run_dir / "checkpoints" / "actor.pt"
        self.model_path.parent.mkdir()
        snapshot = {name: vars(config) for name, config in dict(configs, train=train_config).items()}
        snapshot["train"]["seed"] = seed
        snapshot.update(method=algorithm, algorithm=algorithm, observation=observation)
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8")

        self.swanlab_run = _start_swanlab(train_config, self.run_dir, snapshot)

        self.csv_file, self.csv_writer = _open_episode_csv(self.run_dir, self.episode_fields)


    def log_episode(self, stats, *, episode, global_step, env_id, episode_metadata):
        return _write_training_episode(
            stats, episode_metadata,
            identity=dict(seed=self.seed, episode=episode,
                          global_step=global_step, env_id=int(env_id)),
            csv_file=self.csv_file, csv_writer=self.csv_writer,
            recent_successes=self.recent_successes, swanlab_run=self.swanlab_run)

    def log_learning(self, losses, learn_step):
        if self.swanlab_run is not None:
            self.swanlab_run.log({f"learn/{key}": float(value) for key, value in losses.items()},
                                  step=int(learn_step))

    def close(self):
        self.csv_file.close()
        if self.swanlab_run is not None:
            self.swanlab_run.finish()


class MAPPOLagTrainingLogger(BaselineTrainingLogger):
    episode_fields = BaselineTrainingLogger.episode_fields + ("episode_cost",)
