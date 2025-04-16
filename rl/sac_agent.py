import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models.actor import Actor
from models.critic_classic import ClassicCritic
from models.critic_quantum import QuantumCritic
from rl.replay_buffer import ReplayBuffer

class SACAgent:
    def __init__(
        self, 
        env_info,
        critic_type="classic",
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_entropy=True,
        target_entropy=None,
        buffer_size=1000000,
        quantum_layers=2,
        quantum_wires=4,
        device=torch.device("cpu")
    ):
        """
        Soft Actor-Critic agent with option for quantum critic
        
        Args:
            env_info: Environment information
            critic_type: Type of critic to use ("classic" or "quantum")
            actor_hidden_dims: Hidden layer dimensions for actor
            critic_hidden_dims: Hidden layer dimensions for critic
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            tau: Target network update rate 
            alpha: Entropy regularization coefficient (if auto_entropy is False)
            auto_entropy: Whether to automatically tune alpha
            target_entropy: Target entropy for auto-tuning
            buffer_size: Size of replay buffer
            quantum_layers: Number of quantum layers (if using quantum critic)
            quantum_wires: Number of qubits (if using quantum critic)
            device: Device to run the model on
        """
        self.gamma = gamma
        self.tau = tau
        self.auto_entropy = auto_entropy
        self.device = device
        self.critic_type = critic_type
        
        # Initialize action space parameters
        self.action_dim = env_info.action_dim
        self.action_high = torch.FloatTensor(env_info.action_high).to(device)
        self.action_low = torch.FloatTensor(env_info.action_low).to(device)
        
        # Initialize actor
        self.actor = Actor(
            env_info.observation_dim, 
            self.action_dim,
            self.action_high,
            self.action_low,
            actor_hidden_dims
        ).to(device)
        
        # Initialize critic based on type
        if critic_type == "classic":
            self.critic = ClassicCritic(
                env_info.observation_dim,
                self.action_dim,
                critic_hidden_dims
            ).to(device)
            
            self.critic_target = ClassicCritic(
                env_info.observation_dim,
                self.action_dim,
                critic_hidden_dims
            ).to(device)
        else:  # quantum
            self.critic = QuantumCritic(
                env_info.observation_dim,
                self.action_dim,
                critic_hidden_dims,
                n_qubits=quantum_wires,
                n_layers=quantum_layers,
                device=device
            ).to(device)
            
            self.critic_target = QuantumCritic(
                env_info.observation_dim,
                self.action_dim,
                critic_hidden_dims,
                n_qubits=quantum_wires,
                n_layers=quantum_layers,
                device=device
            ).to(device)
            
        # Initialize target network with same weights
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Set optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize entropy coefficient
        if self.auto_entropy:
            # Target entropy is -dim(A) if not specified
            self.target_entropy = target_entropy if target_entropy is not None else -self.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
        else:
            self.alpha = alpha
            
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            env_info.observation_dim,
            self.action_dim,
            buffer_size,
            device
        )
        
    def select_action(self, state, evaluate=False):
        """
        Select an action from the policy
        
        Args:
            state: Current state
            evaluate: If True, use mean action instead of sampling
            
        Returns:
            Action to take
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if evaluate:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
            
        return action.cpu().numpy()[0]
    
    def update_parameters(self, batch_size):
        """
        Update network parameters using SAC update rule
        
        Args:
            batch_size: Size of the batch to sample from replay buffer
            
        Returns:
            Dictionary of loss values
        """
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        # Current Q estimates
        current_q1, current_q2 = self.critic(states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actions_new, log_probs, _ = self.actor.sample(states)
        q1_new = self.critic.q1_forward(states, actions_new)
        actor_loss = (self.alpha * log_probs - q1_new).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update entropy coefficient if auto-tuning
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            alpha_loss = alpha_loss.item()
        else:
            alpha_loss = 0
            
        # Soft update of target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        # Return metrics
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss,
            'alpha': self.alpha.item() if self.auto_entropy else self.alpha,
            'q_mean': current_q1.mean().item()
        }
        
    def save(self, path):
        """Save model parameters"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy else None
        }, path)
        
    def load(self, path):
        """Load model parameters"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        
        if self.auto_entropy and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp()