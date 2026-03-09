import numpy as np
import matplotlib.pyplot as plt

from example_usage import CooperativeReaching, SEEN_OPPONENTS, UNSEEN_OPPONENTS, ALL_TRAIN_OPPONENTS
from logger import build_probes, evaluate_probes, plot_probe_trends
from dron_moe import DRON_MoE_Agent
from baseline_dqn import BaselineDQNAgent


# =============================================================================
# TRAINING
# =============================================================================

def train_agent(agent, agent_name, opponent_types, num_episodes=5000,
                log_every=200,
                dron_for_probes=None, baseline_for_probes=None,
                probes_seen=None, probes_unseen=None):

    episode_rewards = []
    success_rate    = []

    print(f"\nTraining {agent_name}...")
    if dron_for_probes is not None:
        print(f"Probe logging: every {log_every} episodes")
    print("-" * 60)

    for episode in range(num_episodes):
        opponent_type = np.random.choice(opponent_types)
        env = CooperativeReaching(opponent_type=opponent_type)

        state, opponent_obs = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, opponent_obs, training=True)
            next_state, next_opponent_obs, reward, done = env.step(action)

            if isinstance(agent, DRON_MoE_Agent):
                agent.remember(state, action, next_state, reward, done,
                               opponent_obs, next_opponent_obs)
            else:
                agent.remember(state, action, next_state, reward, done)

            agent.train_step(batch_size=64)

            state        = next_state
            opponent_obs = next_opponent_obs
            episode_reward += reward

        episode_rewards.append(episode_reward)
        success_rate.append(1 if episode_reward > 8 else 0)

        if (episode + 1) % 100 == 0:
            avg_reward  = np.mean(episode_rewards[-100:])
            avg_success = np.mean(success_rate[-100:]) * 100
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Success Rate: {avg_success:.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")

        if (dron_for_probes is not None and baseline_for_probes is not None and
                probes_seen is not None and probes_unseen is not None and
                (episode + 1) % log_every == 0):
            evaluate_probes(
                dron_for_probes, baseline_for_probes,
                probes_seen, probes_unseen, episode + 1
            )

    print(f"\n{agent_name} Training Complete!")
    print(f"Final success rate: {np.mean(success_rate[-100:]) * 100:.1f}%")

    return episode_rewards, success_rate


# =============================================================================
# TESTING
# =============================================================================

def test_agent(agent, agent_name, opponent_types, num_episodes=20):
    results = {}

    print(f"\n{'='*60}")
    print(f"TESTING {agent_name}")
    print(f"{'='*60}")

    for opp_type in opponent_types:
        env       = CooperativeReaching(opponent_type=opp_type)
        successes = 0
        total_rewards = []

        for _ in range(num_episodes):
            state, opponent_obs = env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action = agent.select_action(state, opponent_obs, training=False)
                state, opponent_obs, reward, done = env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)
            if episode_reward > 8:
                successes += 1

        success_rate = successes / num_episodes * 100
        avg_reward   = np.mean(total_rewards)

        results[opp_type] = {
            'success_rate': success_rate,
            'avg_reward':   avg_reward
        }
        print(f"{opp_type:25s}: {success_rate:5.1f}% | Avg Reward: {avg_reward:.2f}")

    overall = np.mean([r['success_rate'] for r in results.values()])
    print(f"\n{agent_name} Overall Average: {overall:.1f}%")

    return results


# =============================================================================
# HOLDOUT EXPERIMENT
# =============================================================================

def holdout_experiment():
    print("\n" + "="*60)
    print("HOLDOUT EXPERIMENT")
    print(f"Train on:  {SEEN_OPPONENTS}")
    print(f"Test on:   {UNSEEN_OPPONENTS} (UNSEEN)")
    print("="*60)

    probes_seen   = build_probes(SEEN_OPPONENTS)
    probes_unseen = build_probes(UNSEEN_OPPONENTS)

    dron_moe = DRON_MoE_Agent(
        state_size=10, opponent_obs_size=6,
        num_actions=4, num_experts=3,
        lr=0.001, gamma=0.95,
        memory_size=10000, target_update_freq=1000, double_dqn=True
    )

    baseline = BaselineDQNAgent(
        state_size=10, num_actions=4,
        lr=0.001, gamma=0.95,
        memory_size=10000, target_update_freq=1000, double_dqn=True
    )

    # ✅ 5000 episodes → 25 probe checkpoints at LOG_EVERY=200
    NUM_EPISODES = 5000
    LOG_EVERY    = 200

    print("\n--- Training DRON-MoE ---")
    dron_rewards, dron_success = train_agent(
        dron_moe, "DRON-MoE", ALL_TRAIN_OPPONENTS,
        num_episodes=NUM_EPISODES, log_every=LOG_EVERY,
        dron_for_probes=dron_moe, baseline_for_probes=baseline,
        probes_seen=probes_seen, probes_unseen=probes_unseen
    )

    print("\n--- Training Baseline DQN ---")
    baseline_rewards, baseline_success = train_agent(
        baseline, "Baseline DQN", ALL_TRAIN_OPPONENTS,
        num_episodes=NUM_EPISODES, log_every=LOG_EVERY,
        dron_for_probes=dron_moe, baseline_for_probes=baseline,
        probes_seen=probes_seen, probes_unseen=probes_unseen
    )

    print("\n--- Testing SEEN ---")
    dron_seen     = test_agent(dron_moe,  "DRON-MoE",    SEEN_OPPONENTS,   num_episodes=20)
    baseline_seen = test_agent(baseline,  "Baseline DQN", SEEN_OPPONENTS,   num_episodes=20)

    print("\n--- Testing UNSEEN ---")
    dron_unseen     = test_agent(dron_moe,  "DRON-MoE",    UNSEEN_OPPONENTS, num_episodes=20)
    baseline_unseen = test_agent(baseline,  "Baseline DQN", UNSEEN_OPPONENTS, num_episodes=20)

    # Summary
    dron_seen_avg       = np.mean([r['success_rate'] for r in dron_seen.values()])
    dron_unseen_avg     = np.mean([r['success_rate'] for r in dron_unseen.values()])
    baseline_seen_avg   = np.mean([r['success_rate'] for r in baseline_seen.values()])
    baseline_unseen_avg = np.mean([r['success_rate'] for r in baseline_unseen.values()])

    print("\n" + "="*60)
    print("HOLDOUT EXPERIMENT SUMMARY")
    print("="*60)
    print(f"\nDRON-MoE:")
    print(f"  Seen opponents:     {dron_seen_avg:.1f}%")
    print(f"  Unseen opponents:   {dron_unseen_avg:.1f}%")
    print(f"  Generalisation gap: {dron_seen_avg - dron_unseen_avg:.1f}%")
    print(f"\nBaseline DQN:")
    print(f"  Seen opponents:     {baseline_seen_avg:.1f}%")
    print(f"  Unseen opponents:   {baseline_unseen_avg:.1f}%")
    print(f"  Generalisation gap: {baseline_seen_avg - baseline_unseen_avg:.1f}%")
    print(f"\nDRON-MoE advantage on unseen: {dron_unseen_avg - baseline_unseen_avg:.1f}%")

    plot_holdout_comparison(
        dron_rewards, baseline_rewards,
        dron_success, baseline_success,
        dron_seen, dron_unseen,
        baseline_seen, baseline_unseen
    )

    print("\n--- Plotting probe trends ---")
    plot_probe_trends()

    return dron_moe, baseline


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def main_comparison():
    print("="*60)
    print("DRON-MoE vs BASELINE DQN — FULL COMPARISON")
    print("="*60)

    all_opponents = [
        'top_left_preference', 'top_right_preference',
        'bottom_left_preference', 'bottom_right_preference',
        'left_preference', 'right_preference'
    ]

    probes_seen   = build_probes(SEEN_OPPONENTS)
    probes_unseen = build_probes(UNSEEN_OPPONENTS)

    print("\n### TRAINING DRON-MoE ###")
    dron_moe = DRON_MoE_Agent(
        state_size=10, opponent_obs_size=6,
        num_actions=4, num_experts=3,
        lr=0.001, gamma=0.95,
        memory_size=10000, target_update_freq=1000, double_dqn=True
    )
    dron_rewards, dron_success = train_agent(
        dron_moe, "DRON-MoE", all_opponents,
        num_episodes=5000, log_every=200,
        dron_for_probes=dron_moe, baseline_for_probes=None,
        probes_seen=probes_seen, probes_unseen=probes_unseen
    )

    print("\n### TRAINING BASELINE DQN ###")
    baseline = BaselineDQNAgent(
        state_size=10, num_actions=4,
        lr=0.001, gamma=0.95,
        memory_size=10000, target_update_freq=1000, double_dqn=True
    )
    baseline_rewards, baseline_success = train_agent(
        baseline, "Baseline DQN", all_opponents,
        num_episodes=5000, log_every=200
    )

    dron_results     = test_agent(dron_moe,  "DRON-MoE",    all_opponents + ['random'], num_episodes=20)
    baseline_results = test_agent(baseline,  "Baseline DQN", all_opponents + ['random'], num_episodes=20)

    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'Opponent Type':<25} {'DRON-MoE':>12} {'Baseline':>12} {'Difference':>12}")
    print("-"*60)

    for opp_type in all_opponents + ['random']:
        dron_sr     = dron_results[opp_type]['success_rate']
        baseline_sr = baseline_results[opp_type]['success_rate']
        diff        = dron_sr - baseline_sr
        print(f"{opp_type:<25} {dron_sr:>11.1f}% {baseline_sr:>11.1f}% {diff:>+11.1f}%")

    dron_avg     = np.mean([r['success_rate'] for r in dron_results.values()])
    baseline_avg = np.mean([r['success_rate'] for r in baseline_results.values()])

    print("-"*60)
    print(f"{'OVERALL AVERAGE':<25} {dron_avg:>11.1f}% {baseline_avg:>11.1f}% "
          f"{dron_avg - baseline_avg:>+11.1f}%")

    plot_comparison(dron_rewards, baseline_rewards, dron_success, baseline_success)
    plot_probe_trends()

    return dron_moe, baseline


# =============================================================================
# PLOTTING
# =============================================================================

def plot_comparison(dron_rewards, baseline_rewards, dron_success, baseline_success):
    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(smooth(dron_rewards),     label='DRON-MoE',    linewidth=2)
    ax1.plot(smooth(baseline_rewards), label='Baseline DQN', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Learning Curve: Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(smooth([s*100 for s in dron_success]),     label='DRON-MoE',    linewidth=2)
    ax2.plot(smooth([s*100 for s in baseline_success]), label='Baseline DQN', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Learning Curve: Success Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dron_moe_vs_baseline.png', dpi=300)
    print("\n📊 Saved: dron_moe_vs_baseline.png")
    plt.show()


def plot_holdout_comparison(dron_rewards, baseline_rewards, dron_success, baseline_success,
                            dron_seen, dron_unseen, baseline_seen, baseline_unseen):
    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    width = 0.35

    ax1.plot(smooth(dron_rewards),     label='DRON-MoE',    linewidth=2, color='#1f77b4')
    ax1.plot(smooth(baseline_rewards), label='Baseline DQN', linewidth=2, color='#ff7f0e')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Training: Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(smooth([s*100 for s in dron_success]),     label='DRON-MoE',    linewidth=2, color='#1f77b4')
    ax2.plot(smooth([s*100 for s in baseline_success]), label='Baseline DQN', linewidth=2, color='#ff7f0e')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Training: Success Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    seen_labels        = list(dron_seen.keys())
    dron_seen_vals     = [dron_seen[k]['success_rate']     for k in seen_labels]
    baseline_seen_vals = [baseline_seen[k]['success_rate'] for k in seen_labels]
    x = np.arange(len(seen_labels))

    ax3.bar(x - width/2, dron_seen_vals,     width, label='DRON-MoE',    alpha=0.8, color='#1f77b4')
    ax3.bar(x + width/2, baseline_seen_vals, width, label='Baseline DQN', alpha=0.8, color='#ff7f0e')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Test: SEEN Opponents')
    ax3.set_xticks(x)
    ax3.set_xticklabels([l.replace('_preference', '').replace('_', ' ')
                         for l in seen_labels], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 110])

    unseen_labels         = list(dron_unseen.keys())
    dron_unseen_vals      = [dron_unseen[k]['success_rate']     for k in unseen_labels]
    baseline_unseen_vals  = [baseline_unseen[k]['success_rate'] for k in unseen_labels]
    x = np.arange(len(unseen_labels))

    bars1 = ax4.bar(x - width/2, dron_unseen_vals,    width, label='DRON-MoE',    alpha=0.8, color='#1f77b4')
    bars2 = ax4.bar(x + width/2, baseline_unseen_vals, width, label='Baseline DQN', alpha=0.8, color='#ff7f0e')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('Test: UNSEEN Opponents (Generalisation Test) ⭐')
    ax4.set_xticks(x)
    ax4.set_xticklabels([l.replace('_preference', '').replace('_', ' ')
                         for l in unseen_labels], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 110])

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.0f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('holdout_experiment.png', dpi=300)
    print("\n📊 Saved: holdout_experiment.png")
    plt.show()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("Choose experiment:")
    print("1. Full comparison (all opponents)")
    print("2. Holdout experiment (generalisation test)")
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
        print("Invalid choice. Running holdout experiment...")
        holdout_experiment()
