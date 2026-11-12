import numpy as np

from .training_env import TrainingEnv


class IPPOEnv(TrainingEnv):
    def reward(self, old_state, info):
        super().reward(old_state, info)
        terms = info["reward_terms"]
        team = sum(terms[key] for key in ("progress", "path", "heading", "time", "terminal"))
        cfg = self.config
        if info["numerical_failure"]:
            local = np.zeros(self.n)
        else:
            coordination = (info["attachment_norms"] / cfg.attachment_limit)**2
            action_change = np.mean(((info["executed_action"] - old_state["action"])
                                     / self.action_high)**2, axis=1)
            local = (-cfg.coordination_weight * coordination * info["elapsed_time"]
                     -cfg.action_change_weight * action_change)
        info["team_reward"] = float(team)
        info["individual_reward"] = local.copy()
        return team + local
