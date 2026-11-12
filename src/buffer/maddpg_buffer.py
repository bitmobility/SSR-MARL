"""FIFO joint replay with uniform sampling."""

import numpy as np


class MADDPGReplayBuffer:
    def __init__(self, param, n_agents, *, seed=2026):
        capacity = param["replay_capacity"]
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(6,)))
        shapes = {
            "obs": (n_agents, param["obs_dim"]),
            "agents": (n_agents, param["agent_state_dim"]),
            "global": (param["global_state_dim"],),
            "action": (n_agents, param["action_dim"]), "reward": (1,),
            "next_obs": (n_agents, param["obs_dim"]),
            "next_agents": (n_agents, param["agent_state_dim"]),
            "next_global": (param["global_state_dim"],),
        }
        self.data = {key: np.zeros((capacity, *shape), dtype=np.float32) for key, shape in shapes.items()}
        for key in ("agent_mask", "next_agent_mask"):
            self.data[key] = np.zeros((capacity, n_agents, 1), dtype=bool)
        for key in ("terminated", "truncated"):
            self.data[key] = np.zeros((capacity, 1), dtype=bool)

    def add_batch(self, obs, state, action, reward, next_obs, next_state, terminated, truncated):
        transition = dict(obs=obs, action=action, next_obs=next_obs,
                          reward=np.asarray(reward).reshape(-1, 1),
                          terminated=np.asarray(terminated).reshape(-1, 1),
                          truncated=np.asarray(truncated).reshape(-1, 1),
                          **state, **{f"next_{key}": value for key, value in next_state.items()})
        for row in range(len(reward)):
            slot = self.position
            for key, values in transition.items():
                self.data[key][slot] = values[row]
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("Replay buffer is empty")
        indices = self.rng.integers(self.size, size=batch_size)
        self.rng.shuffle(indices)
        return {key: values[indices].copy() for key, values in self.data.items()}

    def __len__(self):
        return self.size
