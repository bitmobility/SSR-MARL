"""MADDPG with masked centralized per-vehicle critics."""

from Code.config import parameters

from copy import deepcopy

import numpy as np
import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.action = nn.Linear(hidden_dim, action_dim)
        nn.init.orthogonal_(self.action.weight, gain=0.01)
        nn.init.zeros_(self.action.bias)

    def forward(self, obs):
        return torch.tanh(self.action(self.net(obs)))


class JointQCritic(nn.Module):
    def __init__(self, agent_state_dim, global_state_dim, action_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(agent_state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(3 * hidden_dim + global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        mask = state["agent_mask"].bool()
        inputs = torch.cat((state["agents"], action), dim=-1).masked_fill(~mask, 0)
        encoded = self.encoder(inputs)
        mean = encoded.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1)
        maximum = encoded.masked_fill(~mask, -torch.inf).amax(dim=1)
        context = torch.cat((mean, maximum, state["global"]), dim=-1)
        features = torch.cat((encoded, context[:, None].expand(-1, encoded.shape[1], -1)), dim=-1)
        return self.head(features).masked_fill(~mask, 0)


class MADDPG:
    algorithm = "maddpg"

    def __init__(self, param, *, training=True):
        device = param["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.actor = Actor(param["obs_dim"], param["action_dim"], param["actor_hidden_dim"]).to(self.device)
        if not training:
            self.actor.eval().requires_grad_(False)
            return
        self.critic = JointQCritic(param["agent_state_dim"], param["global_state_dim"],
                                   param["action_dim"], param["critic_hidden_dim"]).to(self.device)
        self.target_actor = deepcopy(self.actor).eval().requires_grad_(False)
        self.target_critic = deepcopy(self.critic).eval().requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=param["a_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=param["c_lr"])
        self.gamma = param["gamma"]
        self.tau = param["tau"]
        self.max_grad_norm = param["max_grad_norm"]
        self.action_l2 = param["action_l2"]
        self.learn_steps = 0

    def _tensor(self, values):
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def choose_action(self, obs):
        return self.actor(self._tensor(obs)).cpu().numpy()

    @torch.no_grad()
    def td_target(self, batch):
        mask = self._tensor(batch["agent_mask"]).bool()
        reward = self._tensor(batch["reward"])
        target = reward[:, None].expand(-1, mask.shape[1], -1).clone()
        bootstrap = np.flatnonzero(~np.asarray(batch["terminated"]).reshape(-1))
        if len(bootstrap):
            state = {key: self._tensor(batch[f"next_{key}"][bootstrap])
                     for key in ("agents", "global", "agent_mask")}
            obs = self._tensor(batch["next_obs"][bootstrap]).masked_fill(~state["agent_mask"].bool(), 0)
            action = self.target_actor(obs)
            target[bootstrap] += self.gamma * self.target_critic(state, action)
        return target.masked_fill(~mask, 0)

    def policy_q(self, obs, state, replay_action):
        mask = state["agent_mask"].bool()
        policy_action = self.actor(obs.masked_fill(~mask, 0))
        values = torch.zeros_like(mask, dtype=obs.dtype)
        for i in range(obs.shape[1]):
            present = mask[:, i, 0]
            if not present.any():
                continue
            local_state = {key: value[present] for key, value in state.items()}
            action = replay_action[present].detach().clone()
            action[:, i] = policy_action[present, i]
            values[present, i] = self.critic(local_state, action)[:, i]
        return values, policy_action

    def learn(self, batch):
        state = {key: self._tensor(batch[key]) for key in ("agents", "global", "agent_mask")}
        mask = state["agent_mask"].bool()
        action = self._tensor(batch["action"])
        target = self.td_target(batch)
        q = self.critic(state, action)
        critic_loss = self._team_mean((q - target).square(), mask)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        self.critic.requires_grad_(False)
        policy_q, policy_action = self.policy_q(self._tensor(batch["obs"]), state, action)
        action_penalty = self._team_mean(policy_action.square().mean(dim=-1, keepdim=True), mask)
        actor_loss = -self._team_mean(policy_q, mask) + self.action_l2 * action_penalty
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)
        self.soft_update()
        self.learn_steps += 1
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(),
                "q_mean": self._team_mean(q.detach(), mask).item(),
                "target_q_mean": self._team_mean(target, mask).item()}

    @staticmethod
    def _team_mean(values, mask):
        return (values.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1)).mean()

    @torch.no_grad()
    def soft_update(self):
        for online, target in ((self.actor, self.target_actor), (self.critic, self.target_critic)):
            for parameter, target_parameter in zip(online.parameters(), target.parameters()):
                target_parameter.lerp_(parameter, self.tau)

    def save_model(self, path):
        torch.save(self.actor.state_dict(), path)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
