import numpy as np


class EpisodeStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0
        self.env_return = 0.0
        self.cbf_return = 0.0
        self.attachment_sum = 0.0
        self.attachment_max = 0.0
        self.path_squared_sum = 0.0
        self.path_max = 0.0
        self.heading_squared_sum = 0.0
        self.duration = 0.0
        self.success = False
        self.collision = False
        self.coordination_violation = False
        self.path_deviation = False
        self.no_forward_reference = False
        self.numerical_failure = False
        self.termination_reason = ""
        self.qp_failures = 0
        self.qp_failure_statuses = set()

    def update(self, env_reward, cbf_reward, info, cbf_info=None):
        self.steps += 1
        self.env_return += float(env_reward)
        self.cbf_return += float(cbf_reward)
        self.attachment_sum += float(np.sqrt(info["mean_squared_attachment_error"]))
        self.attachment_max = max(self.attachment_max, float(info["max_attachment_error"]))
        self.path_squared_sum += info["path_error"] ** 2
        self.path_max = max(self.path_max, info["path_error"])
        self.heading_squared_sum += info["heading_error"] ** 2
        self.duration += float(info["elapsed_time"])
        self.success |= bool(info["success"])
        self.collision |= bool(info["collision_pairs"])
        self.coordination_violation |= bool(info["coordination_violation"])
        self.path_deviation |= bool(info["path_deviation"])
        self.no_forward_reference |= bool(info["no_forward_reference"])
        self.numerical_failure |= bool(info["numerical_failure"])
        self.termination_reason = info["termination_reason"] or ""
        if cbf_info is not None and cbf_info["fallback_used"]:
            self.qp_failures += 1
            self.qp_failure_statuses.add(cbf_info["status"])

    def result(self):
        return {
            "episode_steps": self.steps,
            "env_return": self.env_return,
            "cbf_return": self.cbf_return,
            "attachment_mean": self.attachment_sum / self.steps,
            "attachment_max": self.attachment_max,
            "path_rmse": float(np.sqrt(self.path_squared_sum / self.steps)),
            "path_max": self.path_max,
            "heading_rmse": float(np.sqrt(self.heading_squared_sum / self.steps)),
            "success": int(self.success),
            "collision": int(self.collision),
            "coordination_violation": int(self.coordination_violation),
            "path_deviation": int(self.path_deviation),
            "no_forward_reference": int(self.no_forward_reference),
            "tracking_failure": int(self.path_deviation or self.no_forward_reference),
            "numerical_failure": int(self.numerical_failure),
            "duration": self.duration,
            "termination_reason": self.termination_reason,
            "qp_failures": self.qp_failures,
            "qp_failure_statuses": ";".join(sorted(self.qp_failure_statuses)),
        }


class TestEpisodeStats(EpisodeStats):
    def reset(self):
        super().reset()
        self.cbf_correction_sum = 0.0
        self.cbf_correction_max = 0.0

    def update(self, env_reward, cbf_reward, info, cbf_info):
        super().update(env_reward, cbf_reward, info, cbf_info)
        correction = cbf_info["correction_cost"]
        self.cbf_correction_sum += correction
        self.cbf_correction_max = max(self.cbf_correction_max, correction)

    def result(self):
        return dict(super().result(),
                    cbf_correction_mean=self.cbf_correction_sum / self.steps,
                    cbf_correction_max=self.cbf_correction_max)


class BaselineEpisodeStats(EpisodeStats):
    def update(self, env_reward, info):
        super().update(env_reward, 0.0, info)

    def result(self):
        metrics = super().result()
        for key in ("cbf_return", "qp_failures", "qp_failure_statuses"):
            metrics.pop(key)
        return metrics


class MAPPOLagEpisodeStats(BaselineEpisodeStats):
    def reset(self):
        super().reset()
        self.episode_cost = 0.0

    def update(self, env_reward, info, cost):
        super().update(env_reward, info)
        self.episode_cost += float(cost)

    def result(self):
        return dict(super().result(), episode_cost=self.episode_cost)


def summarize_episodes(rows, group_fields=("group", "formation", "n_agents", "path_type")):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in group_fields)
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, episodes in groups.items():
        result = dict(zip(group_fields, key))
        successful = [row for row in episodes if row["success"]]
        result.update(n_episodes=len(episodes), n_success=len(successful))
        for flag in ("success", "collision", "coordination_violation", "tracking_failure", "numerical_failure"):
            result[f"{flag}_rate"] = float(np.mean([row[flag] for row in episodes]))
        result["timeout_rate"] = float(np.mean([row["termination_reason"] == "time_limit" for row in episodes]))
        for metric in ("env_return", "attachment_mean", "attachment_max", "path_rmse", "path_max",
                       "heading_rmse", "cbf_correction_mean", "cbf_correction_max"):
            result[metric] = float(np.mean([row[metric] for row in episodes]))
        result["success_duration"] = (float(np.mean([row["duration"] for row in successful]))
                                      if successful else None)
        summaries.append(result)
    return summaries
