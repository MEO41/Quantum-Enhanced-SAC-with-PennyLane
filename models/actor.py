import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_high, action_low, hidden_dims=[256, 256]):
        """
        Actor network that outputs a continuous action distribution
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            action_high: Upper bounds of the action space
            action_low: Lower bounds of the action space
            hidden_dims: List of hidden layer dimensions
        """
        super(Actor, self).__init__()
        
        self.action_dim = action_dim
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_scale = torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32).to(device)
        self.action_bias = torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32).to(device)
        #self.action_scale = torch.FloatTensor((action_high - action_low) / 2.0) #those are works in only cpu
        #self.action_bias = torch.FloatTensor((action_high + action_low) / 2.0)
        
        # Build the network
        layers = [nn.Linear(state_dim, hidden_dims[0]), nn.ReLU()]
        
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            
        self.network = nn.Sequential(*layers)
        
        # Output layers for mean and log_std
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], action_dim)
        
        # Initialize output layers with small weights
        self.mean.weight.data.uniform_(-3e-3, 3e-3)
        self.mean.bias.data.uniform_(-3e-3, 3e-3)
        self.log_std.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std.bias.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        """Forward pass to get action distribution parameters"""
        x = self.network(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Constrain log_std for numerical stability
        return mean, log_std
        
    def sample(self, state):
        """Sample action from the distribution and compute log probability"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Use reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Sample with reparameterization
        y_t = torch.tanh(x_t)   # Apply tanh to constrain actions
        
        # Scale and shift actions to match the action space
        action = y_t * self.action_scale + self.action_bias
        
        # Compute log probability, accounting for the tanh transformation
        log_prob = normal.log_prob(x_t)
        
        # Apply correction for tanh transformation
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)