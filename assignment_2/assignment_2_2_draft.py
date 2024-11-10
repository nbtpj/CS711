from stable_baselines3 import DQN
from ale_py import ALEInterface
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.utils import set_random_seed
from gymnasium.wrappers import FrameStack, FlattenObservation

set_random_seed(0)
ale = ALEInterface()
env = TimeLimit(FrameStack(gym.make("ALE/Pong-v5", obs_type="ram", frameskip=1), num_stack=4), 10000)

policy = DQN('MlpPolicy', env, device="cpu", verbose=1)
policy.learn(3000000)
# policy.save('DQN.pth')
#
# policy = REINFORCE(env, device="mps")
# policy.learn(3000000)
# policy.save('REINFORCE.pth')
