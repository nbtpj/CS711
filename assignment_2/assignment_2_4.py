# from __future__ import annotations

import copy
from typing import Any, Tuple, List, SupportsFloat

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from gymnasium.core import ObsType, ActType
from gymnasium.wrappers import TimeLimit
from torch import nn
import torch.distributions as dist
from stable_baselines3 import DQN
from enum import Enum

DEFAULT_GRID_SIZE = 5
DEFAULT_GRID = np.array(
    [
        [0, 1, 0, 9, 5],
        [0, 0, 0, 0, 0],
        [1, 9, 4, 1, 0],
        [0, 0, 2, 0, 1],
        [0, 1, 9, 0, 0],
    ]
)


class CellType(Enum):
    EMPTY = 0
    WALL = 1
    AGENT = 2
    MONSTER_IDLE = 3
    STATION = 4
    PRIZE = 5
    AGENT_MONSTER_IDLE = 6
    AGENT_STATION = 7
    AGENT_PRIZE = 8
    MONSTER_ACTIVE = 9
    AGENT_MONSTER_ACTIVE = 10


class Env(gym.Env):
    CELL_TYPES = {
        0: 'EMPTY',
        1: 'WALL',
        2: 'AGENT',
        3: 'MONSTER IDLE',
        4: 'STATION',
        5: 'PRIZE',
        6: 'AGENT & MONSTER IDLE',  # sayin they can step over :)
        7: 'AGENT & STATION',  # sayin they can step over :)
        8: 'AGENT & PRIZE',  # sayin they can step over :)
        9: 'MONSTER ACTIVE',
        10: 'AGENT & MONSTER ACTIVE',  # sayin they can step over :)
    }
    MOVES = {
        0: np.array([-1, 0]),  # up
        1: np.array([1, 0]),  # down
        2: np.array([0, -1]),  # left
        3: np.array([0, 1]),  # right
    }
    MOVE_NAMES = {
        0: 'UP',  # up
        1: 'DOWN',  # down
        2: 'LEFT',  # left
        3: 'RIGHT',  # right
    }
    MOVE_IN = {
        0: 2,
        3: 6,
        9: 10,
        4: 7,
        5: 8
    }
    MOVE_OUT = {
        2: 0,
        6: 3,
        7: 4,
        8: 5,
        10: 9
    }

    def _add_agent_to(self, old_status):
        if old_status in self.MOVE_IN:
            return self.MOVE_IN[old_status]
        return None

    def _move_agent_from(self, old_status):
        if old_status in self.MOVE_OUT:
            return self.MOVE_OUT[old_status]
        return None

    def __init__(self, grid: np.ndarray = DEFAULT_GRID):
        self.grid_size = grid.shape[-1]
        N_BEING_DAMAGED = 2
        N_CELL_STATUSES = len(self.CELL_TYPES)
        grid_shape = np.ones((self.grid_size, self.grid_size), dtype=np.int8) * N_CELL_STATUSES

        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.MultiDiscrete(nvec=grid_shape),
            "being_damaged": gym.spaces.Discrete(N_BEING_DAMAGED)
        })

        self.action_space = gym.spaces.Discrete(4)
        self._state = None
        self.reset(grid=grid)

    def reset(
            self,
            *,
            grid: np.ndarray = DEFAULT_GRID,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        grid = grid if grid is not None else self.observation_space["grid"].sample()
        self._state = {
            "grid": grid,
            "being_damaged": np.any(grid == 10),
        }
        agent_pos = np.array(np.argwhere(self._state["grid"] == 2)).reshape(-1)
        return copy.deepcopy(self._state), {"agent_pos": agent_pos}

    def dynamic(self, state: ObsType, action: ActType) -> Tuple[ObsType, float, bool, np.ndarray]:
        if isinstance(action, np.ndarray):
            action = action.item()
        new_state = copy.deepcopy(state)
        gained_r = 0
        current_pos = np.isin(state["grid"], [2, 6, 7, 8, 10])
        old_pos = np.array(np.where(current_pos)).reshape(-1)
        new_pos = old_pos + self.MOVES[action]
        is_done = False
        if (np.any(new_pos < 0) or np.any(new_pos >= self.grid_size) or state["grid"][*new_pos] in [1, 2]):
            # blocked, no move
            gained_r -= 1.0
            new_pos = old_pos
        else:
            # not blocked
            mv = self._add_agent_to(state["grid"][*new_pos])
            o_mv = self._move_agent_from(state["grid"][*old_pos])
            assert mv is not None and o_mv is not None, "Can't add agent from {} to {}; mv={}; o_mv={}".format(
                state["grid"][*old_pos], state["grid"][*new_pos], mv, o_mv)
            new_state["grid"][*new_pos] = mv
            new_state["grid"][*old_pos] = o_mv
        if new_state["grid"][*new_pos] == 8:
            gained_r += 100
            is_done = True
        if new_state["grid"][*new_pos] == CellType.AGENT_MONSTER_ACTIVE:
            new_state["being_damaged"] = True
        elif new_state["grid"][*new_pos] == CellType.AGENT_STATION:
            new_state["being_damaged"] = False
        if new_state["being_damaged"]:
            gained_r = -2

        return new_state, gained_r, is_done, new_pos

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        new_state, r, is_done, agent_pos = self.dynamic(self._state, action)
        self._state = new_state
        return copy.deepcopy(new_state), r, is_done, False, {"agent_pos": agent_pos}

    def render(self, mode='human'):

        # Define symbols for each cell type
        SYMBOLS = {
            0: '.',
            1: '#',
            2: 'A',
            3: '.',
            4: 'S',
            5: 'P',
            6: 'AM',
            7: 'AS',
            8: 'AP',
            9: 'M',
            10: 'AMA',
        }
        for row in self._state["grid"]:
            row_symbols = [f"{SYMBOLS[cell]:^5}" for cell in row]
            print(" ".join(row_symbols))


class PartiallyObservable(gym.Wrapper):
    def __init__(self, env: Env, radius: int = 1):
        super().__init__(env)
        self.radius = radius
        self._observation_space = copy.deepcopy(self.env.observation_space)
        clipped_nvec = self.env.observation_space["grid"].nvec[:(radius * 2 + 1), :(radius * 2 + 1)]
        self._observation_space["grid"] = gym.spaces.MultiDiscrete(clipped_nvec)

    def _clip_grid(self, grid: np.ndarray, agent_pos: np.ndarray, radius: int):
        ext_agent_pos = agent_pos + radius
        # pad with wall object
        padded_grid = np.pad(grid,
                             ((radius, radius), (radius, radius)),
                             mode='constant', constant_values=1)
        partial_obs_grid = padded_grid[ext_agent_pos[0] - radius: ext_agent_pos[0] + radius + 1,
                           ext_agent_pos[1] - radius: ext_agent_pos[1] + radius + 1, ]
        return partial_obs_grid

    def step(self, action: ActType):
        obs, reward, truncated, done, info = self.env.step(action)
        obs["grid"] = self._clip_grid(obs["grid"], info["agent_pos"], radius=self.radius)
        return obs, reward, truncated, done, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs["grid"] = self._clip_grid(obs["grid"], info["agent_pos"], radius=self.radius)
        return obs, info


class FeatureQObservable(gym.Wrapper):
    def __init__(self, env: Env, radius: int = 1):
        super().__init__(env)
        self.radius = radius
        self._observation_space = copy.deepcopy(self.env.observation_space)
        clipped_nvec = self.env.observation_space["grid"].nvec[:(radius * 2 + 1), :(radius * 2 + 1)]
        self._observation_space["grid"] = gym.spaces.MultiDiscrete(clipped_nvec)
        self._observation_space["direction2prize"] = gym.spaces.MultiDiscrete([4, 4])
        self._observation_space["direction2station"] = gym.spaces.MultiDiscrete([4, 4])
        self._observation_space["direction2monster"] = gym.spaces.MultiDiscrete([4, 4])

    def extract_features(self, grid: np.ndarray, agent_pos: np.ndarray) -> dict:
        direction2prize = np.array([3, 3])  # default
        direction2station = np.array([3, 3])  # default
        direction2nearest_monster = np.array([3, 3])  # default

        prize_pos = np.argwhere(grid == CellType.PRIZE)
        if prize_pos.size != 0:
            dist = prize_pos - agent_pos
            prize_pos = prize_pos[np.argsort(np.abs(dist[:, 0]) + np.abs(dist[:, 1]))]
            direction2prize = 1 + np.sign(prize_pos[0] - agent_pos)

        station_pos = np.argwhere(grid == CellType.STATION)
        if station_pos.size != 0:
            dist = station_pos - agent_pos
            station_pos = station_pos[np.argsort(np.abs(dist[:, 0]) + np.abs(dist[:, 1]))]
            direction2station = 1 + np.sign(station_pos[0] - agent_pos)

        monster_pos = np.argwhere(grid == CellType.MONSTER_ACTIVE)
        if monster_pos.size != 0:
            dist = monster_pos - agent_pos
            monster_pos = monster_pos[np.argsort(np.abs(dist[:, 0]) + np.abs(dist[:, 1]))]
            direction2station = 1 + np.sign(monster_pos[0] - agent_pos)

        return {
            "direction2prize": direction2prize,
            "direction2station": direction2station,
            "direction2monster": direction2nearest_monster,
        }

    def _clip_grid(self, grid: np.ndarray, agent_pos: np.ndarray, radius: int):
        ext_agent_pos = agent_pos + radius
        # pad with wall object
        padded_grid = np.pad(grid,
                             ((radius, radius), (radius, radius)),
                             mode='constant', constant_values=1)
        partial_obs_grid = padded_grid[ext_agent_pos[0] - radius: ext_agent_pos[0] + radius + 1,
                           ext_agent_pos[1] - radius: ext_agent_pos[1] + radius + 1, ]
        return partial_obs_grid

    def step(self, action: ActType):
        obs, reward, truncated, done, info = self.env.step(action)
        obs["grid"] = self._clip_grid(obs["grid"], info["agent_pos"], radius=self.radius)
        obs.update(self.extract_features(obs["grid"], info["agent_pos"]))
        return obs, reward, truncated, done, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs["grid"] = self._clip_grid(obs["grid"], info["agent_pos"], radius=self.radius)
        obs.update(self.extract_features(obs["grid"], info["agent_pos"]))
        return obs, info


class FlattenEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        _observation_space = []
        self.cvt = lambda x: x
        if isinstance(env.observation_space, gym.spaces.Dict):
            self.ordered_k = list(env.observation_space.keys())
            for k in self.ordered_k:
                v = env.observation_space[k]
                if isinstance(v, gym.spaces.MultiDiscrete):
                    flatten_nvec = v.nvec.reshape(-1)
                    _observation_space.append(flatten_nvec)
                elif isinstance(v, gym.spaces.Discrete):
                    _observation_space.append([v.n])
            _observation_space = np.concatenate(_observation_space, axis=-1)
            self.cvt = lambda x: np.concatenate([np.reshape(x[k], (-1)) for k in self.ordered_k], axis=-1)
            self._observation_space = gym.spaces.MultiDiscrete(nvec=_observation_space)

    def step(self, action: ActType):
        obs, reward, truncated, done, info = self.env.step(action)
        obs = self.cvt(obs)
        return obs, reward, truncated, done, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        obs = self.cvt(obs)
        return obs, info


if __name__ == '__main__':
    from stable_baselines3.common.utils import set_random_seed
    from policy import DQN


    env_1 = TimeLimit(FlattenEnvWrapper(PartiallyObservable(Env())), 100)
    env_2 = TimeLimit(FlattenEnvWrapper(FeatureQObservable(Env())), 100)
    env_3 = TimeLimit(FlattenEnvWrapper(Env()), 100)
    for env, name in zip([env_1, env_2, env_3], ['PARTIALLY_OBSERVABLE', 'FEATURE_Q', 'COMPLETE']):
        for seed in range(5):
            set_random_seed(seed)
            policy = DQN(env, device="mps")
            policy.learn(n_steps=10000, warmup_steps=1000, validation_interval=500, batch_size=32,
                         save_validation_to=f"{name}_{seed}_DQN.txt")
            policy.save(f'{name}_{seed}_DQN.pth')
    # env = TimeLimit(FlattenEnvWrapper(Env()), 100)
    # policy = REINFORCE(env, device="mps")
    # policy.learn(n_steps=1000000, validation_interval=1000, batch_size=256)

# if __name__ == "__main__":
#     import time
#
#
#     # partially observable
#     # env = TimeLimit(FlattenEnvWrapper(PartiallyObservable(Env())), 100)
#
#     # partially observable + Feature Q
#     env = TimeLimit(FlattenEnvWrapper(FeatureQObservable(Env())), 100)
#
#     # fully observable
#     # env = TimeLimit(FlattenEnvWrapper(Env()), 100)
#
#     model = DQN("MlpPolicy", env, verbose=1)
#     model.learn(total_timesteps=100000, log_interval=1)

# s, _ = env.reset()
# while True:
#     env.render()
#     time.sleep(1)
#     # action = 0
#     action = env.action_space.sample()
#     print('-' * 30)
#     print(env.MOVE_NAMES[action])
#     print('-' * 30)
#     s, r, _, _, _ = env.step(action)
