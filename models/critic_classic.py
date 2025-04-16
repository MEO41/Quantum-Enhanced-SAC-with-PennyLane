import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        """
        Classic Q-function approximator implemented as MLP
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: List of hidden layer dimensions
        """
        super(ClassicCritic, self).__init__()
        
        # Build Q1 network
        self.q1_layers = []
        self.q1_layers.append(nn.Linear(state_dim + action_dim, hidden_dims[0]))
        
        for i in range(len(hidden_dims)-1):
            self.q1_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        self.q1_layers.append(nn.Linear(hidden_dims[-1], 1))
        
        # Convert lists to ModuleList
        self.q1_layers = nn.ModuleList(self.q1_layers)
        
        # Build Q2 network (double Q-learning)
        self.q2_layers = []
        self.q2_layers.append(nn.Linear(state_dim + action_dim, hidden_dims[0]))
        
        for i in range(len(hidden_dims)-1):
            self.q2_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        self.q2_layers.append(nn.Linear(hidden_dims[-1], 1))
        
        # Convert lists to ModuleList
        self.q2_layers = nn.ModuleList(self.q2_layers)
        
        # Initialize output layers with small weights
        self.q1_layers[-1].weight.data.uniform_(-3e-3, 3e-3)
        self.q1_layers[-1].bias.data.uniform_(-3e-3, 3e-3)
        self.q2_layers[-1].weight.data.uniform_(-3e-3, 3e-3)
        self.q2_layers[-1].bias.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        """Forward pass to get Q-values"""
        x = torch.cat([state, action], dim=1)
        
        # Q1 forward pass
        q1 = x
        for i in range(len(self.q1_layers) - 1):
            q1 = F.relu(self.q1_layers[i](q1))
        q1 = self.q1_layers[-1](q1)
        
        # Q2 forward pass
        q2 = x
        for i in range(len(self.q2_layers) - 1):
            q2 = F.relu(self.q2_layers[i](q2))
        q2 = self.q2_layers[-1](q2)
        
        return q1, q2
    
    def q1_forward(self, state, action):
        """Forward pass for just Q1 network"""
        x = torch.cat([state, action], dim=1)
        
        q1 = x
        for i in range(len(self.q1_layers) - 1):
            q1 = F.relu(self.q1_layers[i](q1))
        q1 = self.q1_layers[-1](q1)
        
        return q1