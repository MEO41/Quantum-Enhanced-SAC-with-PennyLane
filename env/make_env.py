import gym
import numpy as np
import random
import torch

def make_env(env_name, seed=None):
    """Create and configure the environment."""
    env = gym.make(env_name)
    
    if seed is not None:
        # env.seed(seed) # <-- Gym doesn't use this anymore
        env.reset(seed=seed) # <-- Instead it's uses this
        env.action_space.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    return env

class EnvInfo:
    """Helper class to store environment information."""
    def __init__(self, env):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.observation_dim = env.observation_space.shape[0]
        
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = 1
            self.discrete_actions = True
            self.action_high = env.action_space.n - 1
            self.action_low = 0
        else:  # Continuous action space
            self.action_dim = env.action_space.shape[0]
            self.discrete_actions = False
            self.action_high = env.action_space.high
            self.action_low = env.action_space.low