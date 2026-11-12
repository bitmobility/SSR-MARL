"""Recurrent IPPO with shared local actor and critic."""

from Code.config import parameters

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.lstm = nn.LSTM(param["obs_dim"], param["lstm_hidden_dim"],
                            param["lstm_num_layers"], batch_first=True)
        self.mean_layer = nn.Linear(param["lstm_hidden_dim"], param["action_dim"])
        self.log_std = nn.Parameter(torch.full((param["action_dim"],),
                                               float(np.log(param["initial_std"]))),
                                    requires_grad=param["learn_std"])
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.zeros_(self.mean_layer.bias)

    def distribution(self, obs, hidden=None):
        features, hidden = self.lstm(obs, hidden)
        mean = self.mean_layer(features)
        return Normal(mean, self.log_std.clamp(-20, 2).exp()), hidden

    @staticmethod
    def log_prob(distribution, raw):
        correction = 2 * (np.log(2) - raw - F.softplus(-2 * raw))
        return (distribution.log_prob(raw) - correction).sum(dim=-1)

    def forward(self, obs, hidden=None, deterministic=False):
        distribution, hidden = self.distribution(obs, hidden)
        raw = distribution.mean if deterministic else distribution.rsample()
        return torch.tanh(raw), raw, self.log_prob(distribution, raw), hidden


class LocalCritic(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.lstm = nn.LSTM(param["obs_dim"], param["lstm_hidden_dim"],
                            param["lstm_num_layers"], batch_first=True)
        self.value_layer = nn.Linear(param["lstm_hidden_dim"], 1)

    def forward(self, obs, hidden=None):
        features, hidden = self.lstm(obs, hidden)
        return self.value_layer(features).squeeze(-1), hidden


class IPPO:
    algorithm = "ippo"

    def __init__(self, param, *, training=True):
        device = param["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.actor = Actor(param).to(self.device)
        if not training:
            self.actor.eval().requires_grad_(False)
            return
        self.critic = LocalCritic(param).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=param["a_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=param["c_lr"])
        self.clip_ratio = param["clip_ratio"]
        self.ppo_epochs = param["ppo_epochs"]
        self.minibatch_episodes = param["minibatch_episodes"]
        self.entropy_coef = param["entropy_coef"]
        self.value_coef = param["value_coef"]
        self.max_grad_norm = param["max_grad_norm"]
        self.learn_steps = 0

    def _tensor(self, values):
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def initial_state(self, n_agents):
        shape = (self.actor.lstm.num_layers, n_agents, self.actor.lstm.hidden_size)
        return tuple(torch.zeros(shape, device=self.device) for _ in range(2))

    @torch.no_grad()
    def act(self, obs, actor_state, critic_state):
        sequence = self._tensor(obs).unsqueeze(1)
        action, raw, log_prob, next_actor = self.actor(sequence, actor_state)
        value, next_critic = self.critic(sequence, critic_state)
        outputs = tuple(x.squeeze(1).cpu().numpy() for x in (action, raw, log_prob, value))
        return (*outputs, next_actor, next_critic)

    @torch.no_grad()
    def value(self, obs, critic_state):
        value, _ = self.critic(self._tensor(obs).unsqueeze(1), critic_state)
        return value[:, 0].cpu().numpy()

    @torch.no_grad()
    def choose_action(self, obs, hidden=None, deterministic=False):
        action, _, _, hidden = self.actor(self._tensor(obs).unsqueeze(1), hidden, deterministic)
        return action[:, 0].cpu().numpy(), hidden

    def learn(self, rollout):
        metrics = []
        for _ in range(self.ppo_epochs):
            for batch in rollout.minibatches(self.minibatch_episodes):
                active = torch.as_tensor(batch["active"], device=self.device)
                obs = self._tensor(batch["obs"]).masked_fill(~active[..., None], 0)

                distribution, _ = self.actor.distribution(obs)
                selected = Normal(distribution.loc[active], distribution.scale[active])
                raw = self._tensor(batch["pre_tanh_action"])[active]
                log_prob = self.actor.log_prob(selected, raw)
                old_log_prob = self._tensor(batch["old_log_prob"])[active]
                advantage = self._tensor(batch["advantage"])[active]
                log_ratio = log_prob - old_log_prob
                ratio = log_ratio.exp()
                clipped = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio)
                entropy = -self.actor.log_prob(selected, selected.rsample()).mean()
                actor_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
                actor_loss -= self.entropy_coef * entropy
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value, _ = self.critic(obs)
                critic_loss = F.mse_loss(value[active], self._tensor(batch["return"])[active])
                self.critic_optimizer.zero_grad()
                (self.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                self.learn_steps += 1
                metrics.append((self.learn_steps, {
                    "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(),
                    "entropy": entropy.item(),
                    "approx_kl": ((ratio - 1) - log_ratio).mean().item(),
                    "clip_fraction": ((ratio - 1).abs() > self.clip_ratio).float().mean().item(),
                }))
        return metrics

    def save_model(self, path):
        torch.save(self.actor.state_dict(), path)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
