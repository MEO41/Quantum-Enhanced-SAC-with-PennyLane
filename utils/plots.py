import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def smooth(data, window=10):
    """Apply smoothing to data"""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')

def plot_learning_curves(log_dir, experiment_names, save_path=None):
    """
    Plot learning curves for multiple experiments
    
    Args:
        log_dir: Directory containing the logs
        experiment_names: List of experiment names to plot
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    for name in experiment_names:
        data = pd.read_csv(os.path.join(log_dir, name, 'eval_rewards.csv'))
        steps = data['step'].values
        rewards = data['reward'].values
        smooth_rewards = smooth(rewards)
        plt.plot(steps[len(steps)-len(smooth_rewards):], smooth_rewards, label=name)
    
    plt.xlabel('Environment Steps')
    plt.ylabel('Average Reward')
    plt.title('Evaluation Rewards')
    plt.legend()
    plt.grid(True)
    
    # Plot training losses
    plt.subplot(2, 2, 2)
    for name in experiment_names:
        data = pd.read_csv(os.path.join(log_dir, name, 'critic_loss.csv'))
        steps = data['step'].values
        losses = data['value'].values
        smooth_losses = smooth(losses)
        plt.plot(steps[len(steps)-len(smooth_losses):], smooth_losses, label=name)
    
    plt.xlabel('Environment Steps')
    plt.ylabel('Critic Loss')
    plt.title('Training Critic Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot alpha values
    plt.subplot(2, 2, 3)
    for name in experiment_names:
        data = pd.read_csv(os.path.join(log_dir, name, 'alpha.csv'))
        steps = data['step'].values
        alpha = data['value'].values
        smooth_alpha = smooth(alpha)
        plt.plot(steps[len(steps)-len(smooth_alpha):], smooth_alpha, label=name)
    
    plt.xlabel('Environment Steps')
    plt.ylabel('Alpha Value')
    plt.title('Entropy Coefficient')
    plt.legend()
    plt.grid(True)
    
    # Plot Q-values
    plt.subplot(2, 2, 4)
    for name in experiment_names:
        data = pd.read_csv(os.path.join(log_dir, name, 'q_mean.csv'))
        steps = data['step'].values
        q_values = data['value'].values
        smooth_q = smooth(q_values)
        plt.plot(steps[len(steps)-len(smooth_q):], smooth_q, label=name)
    
    plt.xlabel('Environment Steps')
    plt.ylabel('Average Q-Value')
    plt.title('Q-Value Estimates')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_comparison(classic_results, quantum_results, save_path=None):
    """
    Plot comparison between classic and quantum critics
    
    Args:
        classic_results: Dictionary with classic SAC results
        quantum_results: Dictionary with quantum SAC results
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Compare evaluation rewards
    plt.subplot(2, 3, 1)
    plt.plot(classic_results['steps'], classic_results['eval_rewards'], label='Classic')
    plt.plot(quantum_results['steps'], quantum_results['eval_rewards'], label='Quantum')
    plt.xlabel('Environment Steps')
    plt.ylabel('Average Reward')
    plt.title('Evaluation Rewards')
    plt.legend()
    plt.grid(True)
    
    # Compare training time
    plt.subplot(2, 3, 2)
    plt.bar(['Classic', 'Quantum'], [classic_results['training_time'], quantum_results['training_time']])
    plt.ylabel('Training Time (s)')
    plt.title('Training Time Comparison')
    plt.grid(True)
    
    # Compare inference time
    plt.subplot(2, 3, 3)
    plt.bar(['Classic', 'Quantum'], [classic_results['inference_time'], quantum_results['inference_time']])
    plt.ylabel('Inference Time (ms/step)')
    plt.title('Inference Time Comparison')
    plt.grid(True)
    
    # Compare parameter count
    plt.subplot(2, 3, 4)
    plt.bar(['Classic Actor', 'Classic Critic', 'Quantum Actor', 'Quantum Critic'], 
            [classic_results['actor_params'], classic_results['critic_params'], 
             quantum_results['actor_params'], quantum_results['critic_params']])
    plt.ylabel('Parameter Count')
    plt.title('Model Size Comparison')
    plt.grid(True)
    plt.yscale('log')
    
    # Compare final performance
    plt.subplot(2, 3, 5)
    plt.bar(['Classic', 'Quantum'], [classic_results['final_reward'], quantum_results['final_reward']])
    plt.ylabel('Final Average Reward')
    plt.title('Final Performance')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()