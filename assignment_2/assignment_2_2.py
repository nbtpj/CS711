from policy import DQN, REINFORCE
from ale_py import ALEInterface
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

ale = ALEInterface()
env = TimeLimit(gym.make("ALE/Pong-v5", obs_type="ram"), 2000)
policy = DQN(env, device="mps")
policy.learn(3000000)
policy = REINFORCE(env, device="mps")
policy.learn(3000000)