import os
import argparse
import torch
import numpy as np
import time
from datetime import datetime

from env.make_env import make_env, EnvInfo
from rl.sac_agent import SACAgent
from utils.logger import Logger
from utils.plots import plot_learning_curves, plot_comparison
import config

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate SAC with classic or quantum critic")
    parser.add_argument("--critic", type=str, choices=["classic", "quantum"], default="classic",
                        help="Type of critic to use (classic or quantum)")
    parser.add_argument("--env", type=str, default=config.ENV_NAME,
                        help="Environment name")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed")
    parser.add_argument("--steps", type=int, default=config.MAX_STEPS,
                        help="Maximum number of training steps")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory for saving logs")
    parser.add_argument("--eval_freq", type=int, default=config.EVAL_FREQUENCY,
                        help="Evaluation frequency in steps")
    parser.add_argument("--eval_episodes", type=int, default=config.NUM_EVAL_EPISODES,
                        help="Number of episodes for evaluation")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Disable GPU acceleration")
    parser.add_argument("--test", action="store_true",
                        help="Test mode (no training, load model)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to load model for testing")
    
    return parser.parse_args()

def evaluate(agent, env, num_episodes):
    """Evaluate the agent without exploration"""
    rewards = []
    lengths = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
        rewards.append(episode_reward)
        lengths.append(episode_length)
        
    return rewards, lengths

def train(args):
    """Train SAC agent with classic or quantum critic"""
    # Set up environment
    env = make_env(args.env, args.seed)
    eval_env = make_env(args.env, args.seed + 100)  # Different seed for eval
    env_info = EnvInfo(env)
    
    # Set up device
    device = torch.device("cpu") if args.no_gpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up agent
    agent = SACAgent(
        env_info=env_info,
        critic_type=args.critic,
        actor_hidden_dims=config.ACTOR_HIDDEN_DIMS,
        critic_hidden_dims=config.CRITIC_HIDDEN_DIMS,
        lr_actor=config.LEARNING_RATE_ACTOR,
        lr_critic=config.LEARNING_RATE_CRITIC,
        gamma=config.GAMMA,
        tau=config.TAU,
        alpha=config.ALPHA,
        auto_entropy=config.AUTO_ENTROPY_TUNING,
        target_entropy=config.TARGET_ENTROPY,
        buffer_size=config.REPLAY_BUFFER_SIZE,
        quantum_layers=config.QUANTUM_LAYERS,
        quantum_wires=config.QUANTUM_WIRES,
        device=device
    )
    
    # Set up logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.critic}_sac_{timestamp}"
    logger = Logger(args.log_dir, experiment_name)
    logger.log_model_info(agent)
    
    # Initialize variables
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_num = 0
    
    # Start training
    print(f"Starting training with {args.critic} critic for {args.steps} steps")
    start_time = time.time()
    
    for step in range(1, args.steps + 1):
        # Collect experience
        if step < config.INITIAL_RANDOM_STEPS:
            # Random exploration initially
            action = env.action_space.sample()
        else:
            # Use policy after initial exploration
            action = agent.select_action(state)
        
        # Execute action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        episode_length += 1
        
        # Store transition
        agent.replay_buffer.store(state, action, reward, next_state, done)
        
        # Move to next state
        state = next_state
        
        # Update when enough samples
        if step >= config.UPDATE_AFTER and step % config.GRADIENT_STEPS == 0:
            metrics = agent.update_parameters(config.BATCH_SIZE)
            logger.log_step(step, metrics)
        
        # End of episode handling
        if done:
            logger.log_episode(episode_num, episode_reward, episode_length, step)
            
            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_num += 1
        
        # Evaluation
        if step % args.eval_freq == 0:
            eval_rewards, eval_lengths = evaluate(agent, eval_env, args.eval_episodes)
            mean_reward = logger.log_eval(step, eval_rewards, eval_lengths)
            
            # Save model
            model_dir = os.path.join(args.log_dir, experiment_name, "models")
            os.makedirs(model_dir, exist_ok=True)
            agent.save(os.path.join(model_dir, f"sac_step_{step}.pt"))
            
            # Save best model
            if mean_reward >= max(logger.eval_rewards):
                agent.save(os.path.join(model_dir, "sac_best.pt"))
    
    # Save final model and log training time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    agent.save(os.path.join(args.log_dir, experiment_name, "models", "sac_final.pt"))
    logger.close()
    
    return agent, logger

def test(args):
    """Test a trained agent"""
    # Set up environment
    env = make_env(args.env, args.seed)
    env_info = EnvInfo(env)
    
    # Set up device
    device = torch.device("cpu") if args.no_gpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up agent
    agent = SACAgent(
        env_info=env_info,
        critic_type=args.critic,
        actor_hidden_dims=config.ACTOR_HIDDEN_DIMS,
        critic_hidden_dims=config.CRITIC_HIDDEN_DIMS,
        device=device
    )
    
    # Load model
    agent.load(args.model_path)
    print(f"Loaded model from {args.model_path}")
    
    # Run test episodes
    print(f"Testing agent for {args.eval_episodes} episodes")
    rewards, lengths = evaluate(agent, env, args.eval_episodes)
    
    # Print results
    print(f"Mean reward: {np.mean(rewards):.2f}, Mean length: {np.mean(lengths):.2f}")
    
    # Run and render one episode for visualization
    print("Rendering one episode...")
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        env.render()
        action = agent.select_action(state, evaluate=True)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        time.sleep(0.02)  # Slow down rendering
    
    env.close()
    print(f"Episode reward: {total_reward:.2f}")

if __name__ == "__main__":
    args = parse_args()
    
    if args.test:
        if args.model_path is None:
            raise ValueError("Please provide a model path for testing")
        test(args)
    else:
        train(args)