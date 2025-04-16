import sys
import os
import torch
import numpy as np
import gym

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.actor import Actor
from models.critic_classic import ClassicCritic
from models.critic_quantum import QuantumCritic
from env.make_env import make_env, EnvInfo
from rl.sac_agent import SACAgent
from rl.replay_buffer import ReplayBuffer

def test_env():
    """Test environment setup"""
    print("Testing environment...")
    
    env = make_env("LunarLanderContinuous-v2", seed=42)
    env_info = EnvInfo(env)
    
    print(f"Observation dim: {env_info.observation_dim}")
    print(f"Action dim: {env_info.action_dim}")
    print(f"Action space: {env_info.action_space}")
    print(f"Observation space: {env_info.observation_space}")
    
    # Test reset and step
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print(f"Next state shape: {next_state.shape}, Reward: {reward}, Done: {done}")
    
    print("Environment test passed!\n")
    return env_info

def test_actor(env_info):
    """Test actor network"""
    print("Testing actor network...")
    
    device = torch.device("cpu")
    actor = Actor(
        env_info.observation_dim,
        env_info.action_dim,
        env_info.action_high,
        env_info.action_low,
        [64, 64]
    ).to(device)
    
    # Test forward pass
    state = torch.randn(1, env_info.observation_dim).to(device)
    mean, log_std = actor(state)
    print(f"Actor mean shape: {mean.shape}, log_std shape: {log_std.shape}")
    
    # Test action sampling
    action, log_prob, deterministic_action = actor.sample(state)
    print(f"Action shape: {action.shape}, log_prob shape: {log_prob.shape}")
    print(f"Action range: [{action.min().item():.2f}, {action.max().item():.2f}]")
    
    print("Actor test passed!\n")
    return actor

def test_classic_critic(env_info):
    """Test classic critic network"""
    print("Testing classic critic network...")
    
    device = torch.device("cpu")
    critic = ClassicCritic(
        env_info.observation_dim,
        env_info.action_dim,
        [64, 64]
    ).to(device)
    
    # Test forward pass
    state = torch.randn(1, env_info.observation_dim).to(device)
    action = torch.randn(1, env_info.action_dim).to(device)
    q1, q2 = critic(state, action)
    print(f"Q1 shape: {q1.shape}, Q2 shape: {q2.shape}")
    
    # Test single Q network
    q1_single = critic.q1_forward(state, action)
    print(f"Q1 single shape: {q1_single.shape}")
    
    print("Classic critic test passed!\n")
    return critic

def test_quantum_critic(env_info):
    """Test quantum critic network"""
    print("Testing quantum critic network...")
    
    try:
        import pennylane as qml
        device = torch.device("cpu")
        critic = QuantumCritic(
            env_info.observation_dim,
            env_info.action_dim,
            [64, 64],
            n_qubits=4,
            n_layers=2,
            device=device
        ).to(device)
        
        # Test forward pass
        state = torch.randn(1, env_info.observation_dim).to(device)
        action = torch.randn(1, env_info.action_dim).to(device)
        q1, q2 = critic(state, action)
        print(f"Q1 shape: {q1.shape}, Q2 shape: {q2.shape}")
        
        # Test single Q network
        q1_single = critic.q1_forward(state, action)
        print(f"Q1 single shape: {q1_single.shape}")
        
        print("Quantum critic test passed!\n")
        return critic
    except ImportError:
        print("PennyLane not available, skipping quantum critic test\n")
        return None

def test_replay_buffer(env_info):
    """Test replay buffer"""
    print("Testing replay buffer...")
    
    device = torch.device("cpu")
    buffer = ReplayBuffer(
        env_info.observation_dim,
        env_info.action_dim,
        100,
        device
    )
    
    # Test storing transitions
    for i in range(10):
        state = np.random.randn(env_info.observation_dim)
        action = np.random.randn(env_info.action_dim)
        reward = np.random.rand()
        next_state = np.random.randn(env_info.observation_dim)
        done = False if i < 9 else True
        
        buffer.store(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Test sampling
    batch = buffer.sample(5)
    states, actions, rewards, next_states, dones = batch
    print(f"Batch shapes: States {states.shape}, Actions {actions.shape}, "
          f"Rewards {rewards.shape}, Next states {next_states.shape}, Dones {dones.shape}")
    
    print("Replay buffer test passed!\n")
    return buffer

def test_sac_agent(env_info):
    """Test SAC agent"""
    print("Testing SAC agent...")
    
    device = torch.device("cpu")
    
    # Test classic agent
    classic_agent = SACAgent(
        env_info,
        critic_type="classic",
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        device=device
    )
    
    # Test action selection
    state = np.random.randn(env_info.observation_dim)
    action = classic_agent.select_action(state)
    print(f"Classic agent action: {action}")
    
    try:
        # Test quantum agent if PennyLane is available
        import pennylane as qml
        quantum_agent = SACAgent(
            env_info,
            critic_type="quantum",
            actor_hidden_dims=[64, 64],
            critic_hidden_dims=[64, 64],
            quantum_layers=2,
            quantum_wires=4,
            device=device
        )
        
        # Test action selection
        action = quantum_agent.select_action(state)
        print(f"Quantum agent action: {action}")
        print("Quantum agent test passed!")
    except ImportError:
        print("PennyLane not available, skipping quantum agent test")
    
    print("SAC agent test passed!\n")

def run_sanity_check():
    """Run all tests"""
    print("\n===== RUNNING SANITY CHECKS =====\n")
    
    env_info = test_env()
    actor = test_actor(env_info)
    classic_critic = test_classic_critic(env_info)
    quantum_critic = test_quantum_critic(env_info)
    buffer = test_replay_buffer(env_info)
    test_sac_agent(env_info)
    
    print("All tests passed! System is ready to train.")

if __name__ == "__main__":
    run_sanity_check()