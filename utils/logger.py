import torch
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
import os

class Logger:
    def __init__(self, log_dir, experiment_name):
        """
        Logger for tracking training metrics
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.eval_lengths = []
        
        self.metrics = {}
        self.start_time = time.time()
        
    def log_step(self, step, metrics):
        """Log training metrics per step"""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            
            # Log to tensorboard
            self.writer.add_scalar(f'training/{key}', value, step)
            
    def log_episode(self, episode, episode_reward, episode_length, step):
        """Log metrics after each episode"""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Log to tensorboard
        self.writer.add_scalar('episode/reward', episode_reward, episode)
        self.writer.add_scalar('episode/length', episode_length, episode)
        self.writer.add_scalar('episode/step', step, episode)
        
        # Calculate and log moving averages
        window = min(10, len(self.episode_rewards))
        avg_reward = np.mean(self.episode_rewards[-window:])
        avg_length = np.mean(self.episode_lengths[-window:])
        
        self.writer.add_scalar('episode/avg_reward_10', avg_reward, episode)
        self.writer.add_scalar('episode/avg_length_10', avg_length, episode)
        
        print(f"Episode {episode} - Steps: {step}, Reward: {episode_reward:.2f}, "
              f"Avg10: {avg_reward:.2f}, Length: {episode_length}")
        
    def log_eval(self, step, eval_rewards, eval_lengths):
        """Log evaluation metrics"""
        mean_reward = np.mean(eval_rewards)
        mean_length = np.mean(eval_lengths)
        self.eval_rewards.append(mean_reward)
        self.eval_lengths.append(mean_length)
        
        # Log to tensorboard
        self.writer.add_scalar('eval/mean_reward', mean_reward, step)
        self.writer.add_scalar('eval/mean_length', mean_length, step)
        
        # Log best reward so far
        best_reward = np.max(self.eval_rewards)
        self.writer.add_scalar('eval/best_reward', best_reward, step)
        
        elapsed_time = time.time() - self.start_time
        print(f"Evaluation at step {step} - Mean Reward: {mean_reward:.2f}, "
              f"Mean Length: {mean_length:.2f}, Best: {best_reward:.2f}, "
              f"Time: {elapsed_time:.2f}s")
        
        return mean_reward
    
    def log_model_info(self, agent):
        """Log model architecture and parameters"""
        # Get model parameter counts
        actor_params = sum(p.numel() for p in agent.actor.parameters())
        critic_params = sum(p.numel() for p in agent.critic.parameters())
        
        self.writer.add_text('model/critic_type', agent.critic_type, 0)
        self.writer.add_scalar('model/actor_params', actor_params, 0)
        self.writer.add_scalar('model/critic_params', critic_params, 0)
        self.writer.add_scalar('model/total_params', actor_params + critic_params, 0)
        
    def close(self):
        """Close the logger"""
        self.writer.close()