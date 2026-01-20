# comparison_experiment.py - Compare DRON-MoE vs Baseline DQN (with holdout plotting)

import numpy as np
from dron_moe import DRON_MoE_Agent
from baseline_dqn import BaselineDQNAgent
from example_usage import CooperativeReaching
import matplotlib.pyplot as plt

# =============================================================================
# TRAINING FUNCTION (Works for both agents)
# =============================================================================

def train_agent(agent, agent_name, opponent_types, num_episodes=1000):
    """
    Train an agent (either DRON-MoE or Baseline DQN)
    """
    episode_rewards = []
    success_rate = []
    
    print(f"\nTraining {agent_name}...")
    print("-" * 60)
    
    for episode in range(num_episodes):
        # Random opponent type
        opponent_type = np.random.choice(opponent_types)
        env = CooperativeReaching(opponent_type=opponent_type)
        
        state, opponent_obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, opponent_obs, training=True)
            next_state, next_opponent_obs, reward, done = env.step(action)
            
            # Store and train
            agent.remember(state, action, next_state, reward, opponent_obs)
            agent.train_step(batch_size=64)
            
            state = next_state
            opponent_obs = next_opponent_obs
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        success = 1 if episode_reward > 5 else 0
        success_rate.append(success)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_success = np.mean(success_rate[-100:]) * 100
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Success Rate: {avg_success:.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print(f"\n{agent_name} Training Complete!")
    print(f"Final success rate: {np.mean(success_rate[-100:]) * 100:.1f}%")
    
    return episode_rewards, success_rate


# =============================================================================
# TESTING FUNCTION
# =============================================================================

def test_agent(agent, agent_name, opponent_types, num_episodes=20):
    """
    Test agent against all opponent types
    """
    results = {}
    
    print(f"\n{'='*60}")
    print(f"TESTING {agent_name}")
    print(f"{'='*60}")
    
    for opp_type in opponent_types:
        env = CooperativeReaching(opponent_type=opp_type)
        successes = 0
        total_rewards = []
        
        for episode in range(num_episodes):
            state, opponent_obs = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state, opponent_obs, training=False)
                state, opponent_obs, reward, done = env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            if episode_reward > 5:
                successes += 1
        
        success_rate = successes / num_episodes * 100
        avg_reward = np.mean(total_rewards)
        
        results[opp_type] = {
            'success_rate': success_rate,
            'avg_reward': avg_reward
        }
        
        print(f"{opp_type:25s}: {success_rate:5.1f}% | Avg Reward: {avg_reward:.2f}")
    
    overall_success = np.mean([r['success_rate'] for r in results.values()])
    print(f"\n{agent_name} Overall Average: {overall_success:.1f}%")
    
    return results


# =============================================================================
# HOLDOUT EXPERIMENT (WITH PLOTTING!)
# =============================================================================

def holdout_experiment():
    """
    Train on subset of opponents, test on held-out opponents
    """
    print("\n" + "="*60)
    print("HOLDOUT EXPERIMENT")
    print("Train on: top_left, top_right, bottom_left")
    print("Test on: bottom_right, left_preference, right_preference (UNSEEN)")
    print("="*60)
    
    # Training opponents (seen during training)
    train_opponents = [
        'top_left_preference',
        'top_right_preference',
        'bottom_left_preference'
    ]
    
    # Holdout opponents (NOT seen during training)
    holdout_opponents = [
        'bottom_right_preference',
        'left_preference',
        'right_preference'
    ]
    
    # Train DRON-MoE on subset
    print("\n--- Training DRON-MoE on subset ---")
    dron_moe = DRON_MoE_Agent(
        state_size=10,
        opponent_obs_size=5,
        num_actions=4,
        num_experts=3,
        lr=0.001,
        gamma=0.95
    )
    dron_rewards, dron_success = train_agent(dron_moe, "DRON-MoE (Holdout)", train_opponents, num_episodes=1000)
    
    # Train Baseline on subset
    print("\n--- Training Baseline DQN on subset ---")
    baseline = BaselineDQNAgent(
        state_size=10,
        num_actions=4,
        lr=0.001,
        gamma=0.95
    )
    baseline_rewards, baseline_success = train_agent(baseline, "Baseline DQN (Holdout)", train_opponents, num_episodes=1000)
    
    # Test on SEEN opponents
    print("\n" + "="*60)
    print("TESTING ON SEEN OPPONENTS (trained on these)")
    print("="*60)
    
    dron_seen = test_agent(dron_moe, "DRON-MoE", train_opponents, num_episodes=20)
    baseline_seen = test_agent(baseline, "Baseline DQN", train_opponents, num_episodes=20)
    
    # Test on UNSEEN opponents
    print("\n" + "="*60)
    print("TESTING ON UNSEEN OPPONENTS (never trained on these!)")
    print("="*60)
    
    dron_unseen = test_agent(dron_moe, "DRON-MoE", holdout_opponents, num_episodes=20)
    baseline_unseen = test_agent(baseline, "Baseline DQN", holdout_opponents, num_episodes=20)
    
    # Summary
    print("\n" + "="*60)
    print("HOLDOUT EXPERIMENT SUMMARY")
    print("="*60)
    
    dron_seen_avg = np.mean([r['success_rate'] for r in dron_seen.values()])
    dron_unseen_avg = np.mean([r['success_rate'] for r in dron_unseen.values()])
    baseline_seen_avg = np.mean([r['success_rate'] for r in baseline_seen.values()])
    baseline_unseen_avg = np.mean([r['success_rate'] for r in baseline_unseen.values()])
    
    print(f"\nDRON-MoE:")
    print(f"  Seen opponents:   {dron_seen_avg:.1f}%")
    print(f"  Unseen opponents: {dron_unseen_avg:.1f}%")
    print(f"  Generalization gap: {dron_seen_avg - dron_unseen_avg:.1f}%")
    
    print(f"\nBaseline DQN:")
    print(f"  Seen opponents:   {baseline_seen_avg:.1f}%")
    print(f"  Unseen opponents: {baseline_unseen_avg:.1f}%")
    print(f"  Generalization gap: {baseline_seen_avg - baseline_unseen_avg:.1f}%")
    
    print(f"\nDRON-MoE advantage on unseen opponents: "
          f"{dron_unseen_avg - baseline_unseen_avg:.1f}%")
    
    # Plot holdout results
    plot_holdout_comparison(dron_rewards, baseline_rewards, dron_success, baseline_success,
                           dron_seen, dron_unseen, baseline_seen, baseline_unseen)


# =============================================================================
# MAIN COMPARISON EXPERIMENT
# =============================================================================

def main_comparison():
    """
    Full comparison: DRON-MoE vs Baseline DQN
    """
    print("="*60)
    print("DRON-MoE vs BASELINE DQN COMPARISON")
    print("="*60)
    
    all_opponents = [
        'top_left_preference',
        'top_right_preference',
        'bottom_left_preference',
        'bottom_right_preference',
        'left_preference',
        'right_preference'
    ]
    
    # Train DRON-MoE
    print("\n### TRAINING DRON-MoE ###")
    dron_moe = DRON_MoE_Agent(
        state_size=10,
        opponent_obs_size=5,
        num_actions=4,
        num_experts=3,
        lr=0.001,
        gamma=0.95
    )
    dron_rewards, dron_success = train_agent(
        dron_moe, "DRON-MoE", all_opponents, num_episodes=1000
    )
    
    # Train Baseline
    print("\n### TRAINING BASELINE DQN ###")
    baseline = BaselineDQNAgent(
        state_size=10,
        num_actions=4,
        lr=0.001,
        gamma=0.95
    )
    baseline_rewards, baseline_success = train_agent(
        baseline, "Baseline DQN", all_opponents, num_episodes=1000
    )
    
    # Test both
    dron_results = test_agent(dron_moe, "DRON-MoE", all_opponents + ['random'], num_episodes=20)
    baseline_results = test_agent(baseline, "Baseline DQN", all_opponents + ['random'], num_episodes=20)
    
    # Comparison table
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'Opponent Type':<25} {'DRON-MoE':>12} {'Baseline':>12} {'Difference':>12}")
    print("-"*60)
    
    for opp_type in all_opponents + ['random']:
        dron_sr = dron_results[opp_type]['success_rate']
        baseline_sr = baseline_results[opp_type]['success_rate']
        diff = dron_sr - baseline_sr
        print(f"{opp_type:<25} {dron_sr:>11.1f}% {baseline_sr:>11.1f}% "
              f"{diff:>+11.1f}%")
    
    dron_avg = np.mean([r['success_rate'] for r in dron_results.values()])
    baseline_avg = np.mean([r['success_rate'] for r in baseline_results.values()])
    
    print("-"*60)
    print(f"{'OVERALL AVERAGE':<25} {dron_avg:>11.1f}% {baseline_avg:>11.1f}% "
          f"{dron_avg - baseline_avg:>+11.1f}%")
    
    # Plot learning curves
    plot_comparison(dron_rewards, baseline_rewards, dron_success, baseline_success)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_comparison(dron_rewards, baseline_rewards, dron_success, baseline_success):
    """Plot learning curves comparing both agents (full comparison)"""
    
    # Smooth the curves
    def smooth(data, window=50):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rewards
    ax1.plot(smooth(dron_rewards), label='DRON-MoE', linewidth=2)
    ax1.plot(smooth(baseline_rewards), label='Baseline DQN', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Learning Curve: Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Success rate
    dron_success_smooth = smooth([s*100 for s in dron_success])
    baseline_success_smooth = smooth([s*100 for s in baseline_success])
    
    ax2.plot(dron_success_smooth, label='DRON-MoE', linewidth=2)
    ax2.plot(baseline_success_smooth, label='Baseline DQN', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Learning Curve: Success Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dron_moe_vs_baseline.png', dpi=300)
    print("\n📊 Plots saved as 'dron_moe_vs_baseline.png'")
    plt.show()


def plot_holdout_comparison(dron_rewards, baseline_rewards, dron_success, baseline_success,
                           dron_seen, dron_unseen, baseline_seen, baseline_unseen):
    """Plot results for holdout experiment"""
    
    # Smooth the curves
    def smooth(data, window=50):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Training rewards
    ax1.plot(smooth(dron_rewards), label='DRON-MoE', linewidth=2, color='#1f77b4')
    ax1.plot(smooth(baseline_rewards), label='Baseline DQN', linewidth=2, color='#ff7f0e')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Training: Rewards (on subset of opponents)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top right: Training success rate
    dron_success_smooth = smooth([s*100 for s in dron_success])
    baseline_success_smooth = smooth([s*100 for s in baseline_success])
    
    ax2.plot(dron_success_smooth, label='DRON-MoE', linewidth=2, color='#1f77b4')
    ax2.plot(baseline_success_smooth, label='Baseline DQN', linewidth=2, color='#ff7f0e')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Training: Success Rate (on subset of opponents)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bottom left: Test performance on SEEN opponents
    seen_labels = list(dron_seen.keys())
    dron_seen_vals = [dron_seen[k]['success_rate'] for k in seen_labels]
    baseline_seen_vals = [baseline_seen[k]['success_rate'] for k in seen_labels]
    
    x = np.arange(len(seen_labels))
    width = 0.35
    
    ax3.bar(x - width/2, dron_seen_vals, width, label='DRON-MoE', alpha=0.8, color='#1f77b4')
    ax3.bar(x + width/2, baseline_seen_vals, width, label='Baseline DQN', alpha=0.8, color='#ff7f0e')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Test Performance: SEEN Opponents (trained on these)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([l.replace('_preference', '').replace('_', ' ') for l in seen_labels], 
                         rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 110])
    
    # Bottom right: Test performance on UNSEEN opponents (KEY CHART!)
    unseen_labels = list(dron_unseen.keys())
    dron_unseen_vals = [dron_unseen[k]['success_rate'] for k in unseen_labels]
    baseline_unseen_vals = [baseline_unseen[k]['success_rate'] for k in unseen_labels]
    
    x = np.arange(len(unseen_labels))
    
    bars1 = ax4.bar(x - width/2, dron_unseen_vals, width, label='DRON-MoE', alpha=0.8, color='#1f77b4')
    bars2 = ax4.bar(x + width/2, baseline_unseen_vals, width, label='Baseline DQN', alpha=0.8, color='#ff7f0e')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('Test Performance: UNSEEN Opponents (never trained on!) ⭐')
    ax4.set_xticks(x)
    ax4.set_xticklabels([l.replace('_preference', '').replace('_', ' ') for l in unseen_labels], 
                         rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 110])
    
    # Add value labels on bars for bottom right chart
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}%',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('holdout_experiment.png', dpi=300)
    print("\n📊 Holdout plots saved as 'holdout_experiment.png'")
    plt.show()


# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("Choose experiment:")
    print("1. Full comparison (DRON-MoE vs Baseline on all opponents)")
    print("2. Holdout experiment (train on subset, test on unseen)")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        main_comparison()
    elif choice == '2':
        holdout_experiment()
    elif choice == '3':
        main_comparison()
        print("\n\n")
        holdout_experiment()
    else:
        print("Invalid choice. Running full comparison...")
        main_comparison()
