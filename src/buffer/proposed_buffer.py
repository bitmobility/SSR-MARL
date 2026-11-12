import numpy as np


class ProposedRolloutBuffer:
    def __init__(self, param, n_agents, capacity, num_envs=1):
        if capacity < 1 or num_envs < 1:
            raise ValueError("Rollout length and environment count must be positive")
        self.capacity = capacity
        self.num_envs = num_envs
        self.gamma = param["gamma"]
        self.gae_lambda = param["gae_lambda"]
        shapes = {
            "obs": (n_agents, param["obs_dim"]),
            "agents": (n_agents, param["agent_state_dim"]),
            "global": (param["global_state_dim"],),
            "nominal_action": (n_agents, param["action_dim"]),
            "pre_tanh_action": (n_agents, param["action_dim"]),
            "old_log_prob": (n_agents, 1), "agent_mask": (n_agents, 1),
            "reward": (1,), "old_value": (1,), "next_value": (1,),
            "advantage": (1,), "return": (1,),
        }
        self.data = {key: np.zeros((capacity, num_envs, *shape), dtype=np.float32)
                     for key, shape in shapes.items()}
        for key in ("terminated", "truncated", "valid"):
            self.data[key] = np.zeros((capacity, num_envs, 1), dtype=bool)
        self.clear()

    def add_batch(self, env_ids, obs, state, nominal_action, pre_tanh_action,
                  old_log_prob, reward, value, next_value, terminated, truncated, valid):
        env_ids = np.asarray(env_ids, dtype=int)
        transition = {
            "obs": obs, "agents": state["agents"], "global": state["global"],
            "nominal_action": nominal_action, "pre_tanh_action": pre_tanh_action,
            "old_log_prob": old_log_prob,
            "agent_mask": state["agent_mask"],
        }
        for key, values in dict(reward=reward, old_value=value, next_value=next_value,
                                terminated=terminated, truncated=truncated, valid=valid).items():
            transition[key] = np.asarray(values).reshape(len(env_ids), 1)
        for key, values in self.data.items():
            values[self.size] = 0
            if key in transition:
                values[self.size, env_ids] = transition[key]
        self.size += 1
        self.ready = False

    def compute_returns(self):
        advantage = np.zeros((self.num_envs, 1), dtype=np.float32)
        for t in reversed(range(self.size)):
            valid = self.data["valid"][t, :, 0]
            advantage[~valid] = 0
            value = self.data["old_value"][t, valid]
            terminated = self.data["terminated"][t, valid]
            done = terminated | self.data["truncated"][t, valid]
            delta = (self.data["reward"][t, valid]
                     + self.gamma * ~terminated * self.data["next_value"][t, valid] - value)
            advantage[valid] = delta + self.gamma * self.gae_lambda * ~done * advantage[valid]
            self.data["advantage"][t] = advantage
            self.data["return"][t] = 0
            self.data["return"][t, valid] = advantage[valid] + value
        self.ready = True

    def minibatches(self, batch_size):
        if not self.ready:
            raise ValueError("Compute GAE before sampling")
        valid = self.data["valid"][:self.size, :, 0]
        samples = {key: values[:self.size][valid] for key, values in self.data.items()}
        indices = np.random.permutation(len(self))
        advantage = samples["advantage"]
        mean, std = advantage.mean(), advantage.std()
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            batch = {key: values[selected] for key, values in samples.items()}
            batch["advantage"] = (batch["advantage"] - mean) / (std + 1e-8)
            yield batch

    def clear(self):
        self.size = 0
        self.ready = False

    @property
    def full(self):
        return self.size == self.capacity

    def __len__(self):
        return int(self.data["valid"][:self.size].sum())
