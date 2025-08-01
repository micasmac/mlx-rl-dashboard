#!/usr/bin/env python3
"""
MLX-based Reinforcement Learning Training Script
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError:
    print("MLX not found. This script requires Apple Silicon and MLX.")
    print("Install with: pip install mlx")
    sys.exit(1)

import numpy as np
from tqdm import tqdm


class SimpleAgent:
    """Simple MLX-based agent for demonstration"""
    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.optimizer = optim.Adam(learning_rate=0.001)
        
    def forward(self, x):
        return self.model(x)
    
    def train_step(self, states, rewards):
        """Simple training step - simplified for demo"""
        
        def loss_fn():
            predictions = self.model(states)
            return mx.mean((predictions - rewards) ** 2)
        
        # Use the correct MLX pattern
        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
        loss, grads = loss_and_grad_fn()
        
        # Update model parameters
        self.optimizer.update(self.model, grads)
        
        return float(loss)


def train_agent(episodes=100, output_dir="docs/results"):
    """Main training loop"""
    print(f"Starting MLX RL training for {episodes} episodes...")
    print(f"Output directory: {output_dir}")
    
    agent = SimpleAgent()
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    for episode in tqdm(range(episodes), desc="Training"):
        episode_reward = 0
        episode_length = 0
        episode_loss = 0
        
        for step in range(50):  # Reduced steps for faster demo
            # Simulate state and reward
            state = mx.random.normal([4])
            reward_val = float(mx.random.normal([1]) * 0.1 + mx.sin(step * 0.1))
            
            # Create batch dimensions
            state_batch = mx.expand_dims(state, 0)
            reward_batch = mx.expand_dims(mx.array([reward_val]), 0)
            
            # Training step
            try:
                loss = agent.train_step(state_batch, reward_batch)
                episode_loss += loss
            except Exception as e:
                print(f"Training error at episode {episode}, step {step}: {e}")
                episode_loss += 0.1
            
            episode_reward += reward_val
            episode_length += 1
            
            # Simple termination
            if reward_val > 0.5 or step >= 49:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        losses.append(episode_loss / max(episode_length, 1))
    
    # Save results
    results = {
        'episodes': list(range(len(episode_rewards))),
        'rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'losses': losses,
        'timestamp': datetime.now().isoformat(),
        'total_episodes': episodes,
        'framework': 'MLX',
        'agent_type': 'SimpleAgent'
    }
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    result_file = output_path / "latest_run.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Results saved to {result_file}")
    
    return results


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Train MLX RL Agent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=100,
        help='Number of training episodes'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='docs/results',
        help='Output directory for results'
    )
    
    # Parse arguments
    try:
        args = parser.parse_args()
        print(f"Arguments parsed: episodes={args.episodes}, output={args.output}")
    except SystemExit:
        print("Argument parsing failed. Using defaults.")
        print("Usage: python train.py --episodes 10 --output docs/results")
        return
    
    # Run training
    train_agent(episodes=args.episodes, output_dir=args.output)


if __name__ == "__main__":
    main()
