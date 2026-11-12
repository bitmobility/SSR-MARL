import numpy as np


class EpisodeTrajectory:
    def __init__(self):
        self.steps = []

    def add(self, obs, action, pre_tanh_action, old_log_prob, reward,
            value, next_value, terminated, truncated):
        values = dict(obs=obs, action=action, pre_tanh_action=pre_tanh_action,
                      old_log_prob=old_log_prob, reward=reward, old_value=value,
                      next_value=next_value, terminated=terminated, truncated=truncated)
        self.steps.append({key: np.array(value, copy=True) for key, value in values.items()})

    def arrays(self):
        return {key: np.stack([step[key] for step in self.steps]) for key in self.steps[0]}

    def __len__(self):
        return len(self.steps)


class IPPORolloutBuffer:
    def __init__(self, param):
        self.gamma = param["gamma"]
        self.gae_lambda = param["gae_lambda"]
        self.clear()

    def add_episode(self, episode):
        self.episodes.append(episode.arrays())
        self.ready = False

    def compute_returns(self):
        for episode in self.episodes:
            reward, value = episode["reward"], episode["old_value"]
            advantage = np.zeros_like(reward, dtype=np.float32)
            gae = np.zeros(reward.shape[1], dtype=np.float32)
            for t in reversed(range(len(reward))):
                terminal = episode["terminated"][t]
                done = terminal | episode["truncated"][t]
                delta = reward[t] + self.gamma * (not terminal) * episode["next_value"][t] - value[t]
                gae = delta + self.gamma * self.gae_lambda * (not done) * gae
                advantage[t] = gae
            episode["advantage"] = advantage
            episode["return"] = advantage + value
        self.ready = True

    def minibatches(self, batch_episodes):
        if not self.ready:
            raise ValueError("Compute GAE before sampling")
        advantages = np.concatenate([ep["advantage"].ravel() for ep in self.episodes])
        mean, std = advantages.mean(), advantages.std() + 1e-8
        order = np.random.permutation(len(self.episodes))
        fields = ("obs", "action", "pre_tanh_action", "old_log_prob",
                  "old_value", "advantage", "return")
        for start in range(0, len(order), batch_episodes):
            episodes = [self.episodes[i] for i in order[start:start + batch_episodes]]
            length = max(len(ep["reward"]) for ep in episodes)
            count = sum(ep["reward"].shape[1] for ep in episodes)
            batch = {key: np.zeros((count, length, *episodes[0][key].shape[2:]), dtype=np.float32)
                     for key in fields}
            active = np.zeros((count, length), dtype=bool)
            first = 0
            for ep in episodes:
                t, n = ep["reward"].shape
                active[first:first+n, :t] = True
                for key in fields:
                    batch[key][first:first+n, :t] = ep[key].swapaxes(0, 1)
                first += n
            batch["advantage"][active] = (batch["advantage"][active] - mean) / std
            batch["active"] = active
            yield batch

    def clear(self):
        self.episodes = []
        self.ready = False

    def __len__(self):
        return sum(len(ep["reward"]) for ep in self.episodes)
