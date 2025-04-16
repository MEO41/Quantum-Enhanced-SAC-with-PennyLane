import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size, device):
        """
        Replay buffer for off-policy learning
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            buffer_size: Maximum size of the buffer
            device: Device to store the batches on
        """
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Storage
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
    
    def store(self, state, action, reward, next_state, done):
        """Store a transition in the buffer"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        # Update pointer
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = (
            torch.FloatTensor(self.states[indices]).to(self.device),
            torch.FloatTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_states[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device)
        )
        
        return batch
    
    def __len__(self):
        """Return the current size of the buffer"""
        return self.size