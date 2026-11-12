import numpy as np


class MAPPOLagRolloutBuffer:
    def __init__(self, param, n_agents, capacity, num_envs=1):
        if capacity < 1 or num_envs < 1:
            raise ValueError("Rollout length and environment count must be positive")
        self.capacity = capacity
        self.num_envs = num_envs
        self.gamma = param["gamma"]
        self.gae_lambda = param["gae_lambda"]
        self.cost_gae_lambda = param["cost_gae_lambda"]
        shapes = {
            "obs": (n_agents, param["obs_dim"]),
            "agents": (n_agents, param["agent_state_dim"]),
            "global": (param["global_state_dim"],),
            "action": (n_agents, param["action_dim"]),
            "pre_tanh_action": (n_agents, param["action_dim"]),
            "old_log_prob": (n_agents, 1), "agent_mask": (n_agents, 1),
            "reward": (1,), "old_value": (1,), "next_value": (1,),
            "advantage": (1,), "return": (1,),
            "cost": (1,), "old_cost_value": (1,), "next_cost_value": (1,),
            "cost_advantage": (1,), "cost_return": (1,),
        }
        self.data = {key: np.zeros((capacity, num_envs, *shape), dtype=np.float32)
                     for key, shape in shapes.items()}
        for key in ("terminated", "truncated", "valid"):
            self.data[key] = np.zeros((capacity, num_envs, 1), dtype=bool)
        self.clear()

    def add_batch(self, env_ids, obs, state, action, pre_tanh_action, old_log_prob,
                  reward, cost, value, cost_value, next_value, next_cost_value,
                  terminated, truncated, valid):
        env_ids = np.asarray(env_ids, dtype=int)
        transition = {
            "obs": obs, "agents": state["agents"], "global": state["global"],
            "agent_mask": state["agent_mask"], "action": action,
            "pre_tanh_action": pre_tanh_action, "old_log_prob": old_log_prob,
        }
        for key, values in dict(reward=reward, cost=cost, old_value=value,
                                old_cost_value=cost_value, next_value=next_value,
                                next_cost_value=next_cost_value, terminated=terminated,
                                truncated=truncated, valid=valid).items():
            transition[key] = np.asarray(values).reshape(len(env_ids), 1)
        for key, values in self.data.items():
            values[self.size] = 0
            if key in transition:
                values[self.size, env_ids] = transition[key]
        self.size += 1
        self.ready = False

    def compute_returns(self):
        advantage = np.zeros((self.num_envs, 1), dtype=np.float32)
        cost_advantage = np.zeros_like(advantage)
        for t in reversed(range(self.size)):
            valid = self.data["valid"][t, :, 0]
            advantage[~valid] = 0
            cost_advantage[~valid] = 0
            terminated = self.data["terminated"][t, valid]
            done = terminated | self.data["truncated"][t, valid]
            value = self.data["old_value"][t, valid]
            cost_value = self.data["old_cost_value"][t, valid]
            delta = (self.data["reward"][t, valid]
                     + self.gamma * ~terminated * self.data["next_value"][t, valid] - value)
            cost_delta = (self.data["cost"][t, valid]
                          + ~done * self.data["next_cost_value"][t, valid] - cost_value)
            advantage[valid] = delta + self.gamma * self.gae_lambda * ~done * advantage[valid]
            cost_advantage[valid] = cost_delta + self.cost_gae_lambda * ~done * cost_advantage[valid]
            self.data["advantage"][t] = advantage
            self.data["cost_advantage"][t] = cost_advantage
            self.data["return"][t] = 0
            self.data["cost_return"][t] = 0
            self.data["return"][t, valid] = advantage[valid] + value
            self.data["cost_return"][t, valid] = cost_advantage[valid] + cost_value
        self.ready = True

    def minibatches(self, batch_size):
        if not self.ready:
            raise ValueError("Compute GAE before sampling")
        valid = self.data["valid"][:self.size, :, 0]
        samples = {key: values[:self.size][valid] for key, values in self.data.items()}
        indices = np.random.permutation(len(self))
        mean, std = samples["advantage"].mean(), samples["advantage"].std()
        cost_mean = samples["cost_advantage"].mean()
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            batch = {key: values[selected] for key, values in samples.items()}
            batch["advantage"] = (batch["advantage"] - mean) / (std + 1e-8)

            batch["cost_advantage"] -= cost_mean
            yield batch

    def clear(self):
        self.size = 0
        self.ready = False

    @property
    def full(self):
        return self.size == self.capacity

    def __len__(self):
        return int(self.data["valid"][:self.size].sum())
