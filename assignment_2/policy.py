import copy
import math
from typing import List, Tuple
import numpy as np
import torch.nn.functional as F
import torch
import torch.distributions as dist
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
from torch import nn
import gymnasium as gym
from gymnasium import spaces
from tqdm import trange


def obs2tensor(x: np.array, space: spaces.Space) -> torch.Tensor:
    if isinstance(space, spaces.MultiDiscrete):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).long()
        op = []
        for i, value in enumerate(range(x.size(-1))):
            # Get the range for the current discrete variable
            num_options = space.nvec[i]
            op.append(F.one_hot(x[..., i], num_classes=num_options))
        ts = torch.cat(op, dim=-1)
    elif isinstance(space, spaces.Discrete):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).long()
        ts = F.one_hot(x, num_classes=space.n)
    else:
        ts = x
        if not torch.is_tensor(x):
            ts = torch.from_numpy(x).float()

    return ts


def get_dim(space: spaces.Space) -> int:
    if isinstance(space, spaces.Discrete):
        return space.n
    elif isinstance(space, spaces.MultiDiscrete):
        flatten_nvec = space.nvec.reshape(-1)
        return sum(flatten_nvec)
    return space.shape[-1]


@torch.no_grad()
def validate_policy(actor, env: gym.Env, n_episodes: int = 5):
    total_return = 0
    for i in trange(n_episodes):
        obs, info = env.reset()
        for j in range(1000):
            action, log_prob = actor.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_return += reward
            if done or truncated:
                break
    return total_return / n_episodes


class DQN:
    def __init__(self, env, device='mps', gamma=0.99, interval_Q_update: int = 100):
        self.env = env
        self.eval_env = copy.deepcopy(env)
        self.gamma = gamma
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.device = device
        self.interval_Q_update = interval_Q_update
        assert isinstance(self.action_space, spaces.Discrete)
        obs_dim = get_dim(self.env.observation_space)
        act_dim = get_dim(self.env.action_space)
        self.Q = nn.Sequential(nn.Linear(obs_dim, 64, ), nn.ReLU(),
                               nn.Linear(64, 64, ), nn.ReLU(),
                               nn.Linear(64, act_dim), ).to(self.device)
        self.target_Q = copy.deepcopy(self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=3e-4)
        self.buffer = ReplayBuffer(int(1e6), self.observation_space, self.action_space, n_envs=1)

    def update(self, batch_size):
        (obs, act, next_obs, done, reward) = self.buffer.sample(batch_size)
        obs = obs2tensor(obs, self.observation_space).to(self.device).float()
        act = act.reshape(-1)
        next_obs = obs2tensor(next_obs, self.observation_space).to(self.device).float()
        reward, done = (reward).to(self.device).reshape(-1), (done).to(self.device).float().reshape(-1)
        reward = (reward - reward.mean()) / (reward.std() + 1e-6)
        with (torch.no_grad()):
            nx_Q = self.target_Q(next_obs)
            target_q_values = reward + self.gamma * (1 - done) * torch.amax(nx_Q, dim=-1, keepdim=False)
        current_Q = self.Q(obs)[np.arange(batch_size), act]
        q_loss = F.mse_loss(current_Q, target_q_values)
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()
        return q_loss.item()

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic=True) -> Tuple[np.ndarray, np.ndarray]:
        obs_ts = obs2tensor(obs, self.observation_space).to(self.device).float()
        logits = self.Q(obs_ts)
        distribution = dist.Categorical(logits=logits)

        if deterministic:
            act = torch.argmax(self.Q(obs_ts), dim=-1)
        else:
            act = distribution.sample().detach()
        log_prob = distribution.log_prob(act)
        return act.cpu().numpy(), log_prob.cpu().numpy()

    def learn(self, n_steps: int = 10000000, validation_interval: int = 5000,
              batch_size=64, warmup_steps: int = 25000, n_episodes: int = 5):
        obs, info = self.env.reset()
        validation_returns = []
        for i in trange(n_steps):
            if i + 1 < warmup_steps:
                action = self.action_space.sample()
            else:
                action, log_prob = self.act(obs, deterministic=False)  # Thomson Sampling
            next_obs, reward, done, truncated, info = self.env.step(action)
            self.buffer.add(obs=obs, next_obs=next_obs, reward=reward, done=done, infos=[info], action=action)
            q_loss = self.update(batch_size=batch_size)
            obs = next_obs
            if (i + 1) % validation_interval == 0:
                eval_return = validate_policy(self, self.eval_env, n_episodes=n_episodes)
                print(f"Q loss = {q_loss:.4f}")
                print(f"J (pi) = {eval_return:.4f}")
                validation_returns.append(eval_return)
                np.savetxt(f'DQN.txt', validation_returns)
            if (i + 1) % self.interval_Q_update == 0:
                self.target_Q.load_state_dict(self.Q.state_dict())

    def save(self, path):
        torch.save(self.Q.state_dict(), path)

    def load(self, path):
        self.Q.load_state_dict(torch.load(path, map_location=self.device))


class REINFORCE:
    def __init__(self, env, device='mps', gamma=0.99, collect_step: int = int(512), K_epoch: int = 2):
        self.env = env
        self.eval_env = copy.deepcopy(env)
        self.gamma = gamma
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.device = device
        self.collect_step = collect_step
        self.K_epoch = K_epoch
        assert isinstance(self.action_space, spaces.Discrete)
        obs_dim = get_dim(self.env.observation_space)
        act_dim = get_dim(self.env.action_space)
        self.pi = nn.Sequential(nn.Linear(obs_dim, 64, ), nn.ReLU(),
                                nn.Linear(64, 64, ), nn.ReLU(),
                                nn.Linear(64, act_dim), ).to(self.device)
        self.optimizer = torch.optim.Adam(self.pi.parameters(), lr=3e-4)

    def forward(self, obs_ts: torch.Tensor):
        logits = self.pi(obs_ts)
        distribution = dist.Categorical(logits=logits)
        action = distribution.sample()
        return action, distribution.log_prob(action)

    def log_prob(self, obs_ts: torch.Tensor, act: torch.Tensor):
        logits = self.pi(obs_ts)
        distribution = dist.Categorical(logits=logits)
        return distribution.log_prob(act)

    def update(self, obs_ts: torch.Tensor, act: torch.Tensor, R: torch.Tensor):
        log_prob = self.log_prob(obs_ts, act)
        pi_loss = (-log_prob * R).mean()
        self.optimizer.zero_grad()
        pi_loss.backward()
        self.optimizer.step()
        return pi_loss.item()

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic=True) -> Tuple[np.ndarray, np.ndarray]:
        obs_ts = obs2tensor(obs, self.observation_space).to(self.device).float()
        logits = self.pi(obs_ts)
        distribution = dist.Categorical(logits=logits)

        if deterministic:
            act = torch.argmax(self.pi(obs_ts), dim=-1)
        else:
            act = distribution.sample().detach()
        log_prob = distribution.log_prob(act)
        return act.cpu().numpy(), log_prob.cpu().numpy()

    def learn(self, n_steps: int = 10000000, validation_interval: int = 5000,
              batch_size=64, n_episodes: int = 5):
        obs, info = self.env.reset()
        validation_returns = []
        rollout_buffer = {
            "obs": [],
            "act": [],
            "r": [],
            "done": []
        }
        pi_loss = 0

        for i in trange(n_steps):
            action, log_prob = self.act(obs, deterministic=False)  # Thomson Sampling
            next_obs, reward, done, truncated, info = self.env.step(action)
            rollout_buffer["obs"].append(obs2tensor(obs, self.observation_space).float())
            rollout_buffer["act"].append(torch.tensor(action).long())
            rollout_buffer["r"].append(torch.tensor(reward).float())
            rollout_buffer["done"].append(torch.tensor(done).float())
            obs = next_obs

            # q_loss = self.update(batch_size=batch_size)
            if (i + 1) % self.collect_step == 0:
                pi_loss = []
                # compute nR (MCMC)
                R = []
                last_R = 0
                for i in reversed(range(len(rollout_buffer["r"]))):
                    last_R = rollout_buffer["r"][i] + self.gamma * last_R * (1 - rollout_buffer["done"][i])
                    R.append(last_R)
                rollout_buffer = {k: torch.stack(v, dim=0).to(self.device) for k, v in rollout_buffer.items()}
                rollout_buffer["R"] = torch.tensor(R).to(self.device).float()
                for ep in trange(self.K_epoch):
                    indices = np.arange(rollout_buffer["R"].shape[0])
                    np.random.shuffle(indices)
                    for batch_ids in torch.split(torch.tensor(indices), batch_size):
                        _pi_loss = self.update(rollout_buffer["obs"][batch_ids], rollout_buffer["act"][batch_ids],
                                               rollout_buffer["R"][batch_ids])
                        pi_loss.append(_pi_loss)
                # reset buffer
                rollout_buffer = {
                    "obs": [],
                    "act": [],
                    "r": [],
                    "done": []
                }
                obs, info = self.env.reset()
            if (i + 1) % validation_interval == 0:
                eval_return = validate_policy(self, self.eval_env, n_episodes=n_episodes)
                validation_returns.append(eval_return)
                print(f"Pi loss = {np.mean(pi_loss):.4f}")
                print(f"J (pi) = {eval_return:.4f}")
                np.savetxt(f'REINFORCE.txt', validation_returns)

    def save(self, path):
        torch.save(self.pi.state_dict(), path)

    def load(self, path):
        self.pi.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == '__main__':
    from ale_py import ALEInterface

    ale = ALEInterface()
    env = TimeLimit(gym.make("ALE/Pong-v5", obs_type="ram"), 2000)
    # policy = DQN(env, device="mps")
    policy = REINFORCE(env, device="mps")
    policy.learn(20)
    policy.save('REINFORCE.pth')
    policy.load('REINFORCE.pth')
