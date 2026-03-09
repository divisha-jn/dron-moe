import os
import numpy as np
import matplotlib.pyplot as plt

from example_usage import CooperativeReaching, SEEN_OPPONENTS, UNSEEN_OPPONENTS
from logger_policy import (
    build_probes, evaluate_probes,
    plot_probe_trends, smooth_xy
)
from dron_moe_policy import DRON_MoE_PolicyAgent
from baseline_policy import BaselinePolicyAgent


# =============================================================================
# TRAINING
# =============================================================================

def train_agent(agent, agent_name, opponent_types,
                num_episodes=5000,
                log_every=200,
                dron_for_probes=None, baseline_for_probes=None,
                probes_seen=None, probes_unseen=None):

    episode_rewards = []
    success_rate    = []
    actor_losses    = []
    critic_losses   = []
    entropies       = []

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
            agent.remember(reward, done, next_state, next_opponent_obs)
            state        = next_state
            opponent_obs = next_opponent_obs
            episode_reward += reward

        # Train at end of every episode
        loss_dict = agent.train_step()
        if loss_dict is not None:
            actor_losses.append(loss_dict.get("actor_loss", 0.0))
            critic_losses.append(loss_dict.get("critic_loss", 0.0))
            entropies.append(loss_dict.get("entropy", 0.0))

        episode_rewards.append(episode_reward)
        success_rate.append(1 if episode_reward > 8 else 0)

        if (episode + 1) % 100 == 0:
            avg_reward      = np.mean(episode_rewards[-100:])
            avg_success     = np.mean(success_rate[-100:]) * 100
            avg_actor_loss  = np.mean(actor_losses[-20:])  if len(actor_losses)  >= 20 else 0
            avg_critic_loss = np.mean(critic_losses[-20:]) if len(critic_losses) >= 20 else 0
            avg_entropy     = np.mean(entropies[-20:])     if len(entropies)     >= 20 else 0

            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Reward: {avg_reward:6.2f} | Success: {avg_success:5.1f}%")
            print(f"  Actor Loss: {avg_actor_loss:6.3f} | "
                  f"Critic Loss: {avg_critic_loss:6.3f} | "
                  f"Entropy: {avg_entropy:5.3f}")

        if (dron_for_probes is not None and baseline_for_probes is not None and
                probes_seen is not None and probes_unseen is not None and
                (episode + 1) % log_every == 0):
            evaluate_probes(
                dron_for_probes, baseline_for_probes,
                probes_seen, probes_unseen, episode + 1
            )

    print(f"\n{agent_name} Training Complete!")
    print(f"Final success rate: {np.mean(success_rate[-100:]) * 100:.1f}%")

    return {
        "rewards":       episode_rewards,
        "success":       success_rate,
        "actor_losses":  actor_losses,
        "critic_losses": critic_losses,
        "entropies":     entropies
    }


# =============================================================================
# TESTING
# =============================================================================

def test_agent(agent, agent_name, opponent_types, num_episodes=50):
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

        results[opp_type] = {"success_rate": success_rate, "avg_reward": avg_reward}
        print(f"{opp_type:25s}: {success_rate:5.1f}% | Avg Reward: {avg_reward:.2f}")

    overall = np.mean([r["success_rate"] for r in results.values()])
    print(f"\n{agent_name} Overall Average: {overall:.1f}%")

    return results


# =============================================================================
# HOLDOUT EXPERIMENT
# =============================================================================

def holdout_experiment():
    print("\n" + "="*60)
    print("HOLDOUT EXPERIMENT — POLICY")
    print(f"Train on:  {SEEN_OPPONENTS}")
    print(f"Test on:   {UNSEEN_OPPONENTS} (UNSEEN)")
    print("="*60)

    probes_seen   = build_probes(SEEN_OPPONENTS)
    probes_unseen = build_probes(UNSEEN_OPPONENTS)

    # Clean old probe CSVs
    for f in ["probe_policy_dron.csv", "probe_policy_baseline.csv"]:
        if os.path.exists(f):
            os.remove(f)

    dron_moe = DRON_MoE_PolicyAgent(
        state_size=10, opponent_obs_size=6,
        num_actions=4, num_experts=3,
        lr=0.00005, gamma=0.99,
        eps_clip=0.2, epochs=4,
        value_coef=0.5, entropy_coef=0.05
    )

    baseline = BaselinePolicyAgent(
        state_size=10, opponent_obs_size=6,
        num_actions=4,
        lr=0.00005, gamma=0.99,
        eps_clip=0.2, epochs=4,
        value_coef=0.5, entropy_coef=0.02
    )

    NUM_EPISODES = 5000
    LOG_EVERY    = 200

    print("\n--- Training DRON-MoE-Policy ---")
    dron_metrics = train_agent(
        dron_moe, "DRON-MoE-Policy", SEEN_OPPONENTS,
        num_episodes=NUM_EPISODES, log_every=LOG_EVERY,
        dron_for_probes=dron_moe, baseline_for_probes=baseline,
        probes_seen=probes_seen, probes_unseen=probes_unseen
    )

    print("\n--- Training Baseline Policy ---")
    baseline_metrics = train_agent(
        baseline, "Baseline Policy", SEEN_OPPONENTS,
        num_episodes=NUM_EPISODES, log_every=LOG_EVERY,
        dron_for_probes=dron_moe, baseline_for_probes=baseline,
        probes_seen=probes_seen, probes_unseen=probes_unseen
    )

    print("\n--- Testing SEEN ---")
    dron_seen     = test_agent(dron_moe,  "DRON-MoE-Policy", SEEN_OPPONENTS,   num_episodes=50)
    baseline_seen = test_agent(baseline,  "Baseline Policy",  SEEN_OPPONENTS,   num_episodes=50)

    print("\n--- Testing UNSEEN ---")
    dron_unseen     = test_agent(dron_moe,  "DRON-MoE-Policy", UNSEEN_OPPONENTS, num_episodes=50)
    baseline_unseen = test_agent(baseline,  "Baseline Policy",  UNSEEN_OPPONENTS, num_episodes=50)

    # Summary
    dron_seen_avg       = np.mean([r['success_rate'] for r in dron_seen.values()])
    dron_unseen_avg     = np.mean([r['success_rate'] for r in dron_unseen.values()])
    baseline_seen_avg   = np.mean([r['success_rate'] for r in baseline_seen.values()])
    baseline_unseen_avg = np.mean([r['success_rate'] for r in baseline_unseen.values()])

    print("\n" + "="*60)
    print("HOLDOUT EXPERIMENT SUMMARY")
    print("="*60)
    print(f"\nDRON-MoE-Policy:")
    print(f"  Seen opponents:     {dron_seen_avg:.1f}%")
    print(f"  Unseen opponents:   {dron_unseen_avg:.1f}%")
    print(f"  Generalisation gap: {dron_seen_avg - dron_unseen_avg:.1f}%")
    print(f"\nBaseline Policy:")
    print(f"  Seen opponents:     {baseline_seen_avg:.1f}%")
    print(f"  Unseen opponents:   {baseline_unseen_avg:.1f}%")
    print(f"  Generalisation gap: {baseline_seen_avg - baseline_unseen_avg:.1f}%")
    print(f"\nDRON-MoE advantage on unseen: {dron_unseen_avg - baseline_unseen_avg:.1f}%")

    plot_final_comparison(
        dron_metrics, baseline_metrics,
        dron_seen, dron_unseen,
        baseline_seen, baseline_unseen
    )

    print("\n--- Plotting probe trends ---")
    plot_probe_trends()

    return dron_moe, baseline


# =============================================================================
# PLOTS
# =============================================================================

def plot_final_comparison(dron_metrics, baseline_metrics,
                          dron_seen, dron_unseen,
                          baseline_seen, baseline_unseen):

    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    fig = plt.figure(figsize=(16, 12))
    gs  = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Training rewards
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(smooth(dron_metrics['rewards']),     label='DRON-MoE-Policy', linewidth=2, color='#1f77b4')
    ax1.plot(smooth(baseline_metrics['rewards']), label='Baseline Policy', linewidth=2, color='#ff7f0e')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Training: Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)

    # Training success rate
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(smooth([s*100 for s in dron_metrics['success']]),     label='DRON-MoE-Policy', linewidth=2, color='#1f77b4')
    ax2.plot(smooth([s*100 for s in baseline_metrics['success']]), label='Baseline Policy',  linewidth=2, color='#ff7f0e')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Training: Success Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    # Actor loss
    ax3 = fig.add_subplot(gs[1, 0])
    if dron_metrics['actor_losses'] and baseline_metrics['actor_losses']:
        ax3.plot(smooth(dron_metrics['actor_losses']),     label='DRON-MoE Actor', linewidth=2, color='#1f77b4')
        ax3.plot(smooth(baseline_metrics['actor_losses']), label='Baseline Actor',  linewidth=2, color='#ff7f0e')
        ax3.set_xlabel('Update Step')
        ax3.set_ylabel('Actor Loss')
        ax3.set_title('Actor Loss Convergence')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Critic loss
    ax4 = fig.add_subplot(gs[1, 1])
    if dron_metrics['critic_losses'] and baseline_metrics['critic_losses']:
        ax4.plot(smooth(dron_metrics['critic_losses']),     label='DRON-MoE Critic', linewidth=2, color='#1f77b4')
        ax4.plot(smooth(baseline_metrics['critic_losses']), label='Baseline Critic',  linewidth=2, color='#ff7f0e')
        ax4.set_xlabel('Update Step')
        ax4.set_ylabel('Critic Loss')
        ax4.set_title('Critic Loss Convergence')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Seen bar chart
    width       = 0.35
    seen_labels = list(dron_seen.keys())
    x           = np.arange(len(seen_labels))

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.bar(x - width/2, [dron_seen[k]['success_rate']     for k in seen_labels],
            width, label='DRON-MoE-Policy', alpha=0.8, color='#1f77b4')
    ax5.bar(x + width/2, [baseline_seen[k]['success_rate'] for k in seen_labels],
            width, label='Baseline Policy',  alpha=0.8, color='#ff7f0e')
    ax5.set_ylabel('Success Rate (%)')
    ax5.set_title('Test: SEEN Opponents')
    ax5.set_xticks(x)
    ax5.set_xticklabels([l.replace('_preference', '').replace('_', '\n')
                         for l in seen_labels], rotation=0, ha='center')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim([0, 110])

    # Unseen bar chart
    unseen_labels = list(dron_unseen.keys())
    x             = np.arange(len(unseen_labels))

    ax6   = fig.add_subplot(gs[2, 1])
    bars1 = ax6.bar(x - width/2, [dron_unseen[k]['success_rate']     for k in unseen_labels],
                    width, label='DRON-MoE-Policy', alpha=0.8, color='#1f77b4')
    bars2 = ax6.bar(x + width/2, [baseline_unseen[k]['success_rate'] for k in unseen_labels],
                    width, label='Baseline Policy',  alpha=0.8, color='#ff7f0e')
    ax6.set_ylabel('Success Rate (%)')
    ax6.set_title('Test: UNSEEN Opponents (Generalisation Test) ⭐')
    ax6.set_xticks(x)
    ax6.set_xticklabels([l.replace('_preference', '').replace('_', '\n')
                         for l in unseen_labels], rotation=0, ha='center')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_ylim([0, 110])

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.0f}%', ha='center', va='bottom', fontsize=9)

    plt.savefig('holdout_experiment_policy.png', dpi=300)
    print("\n📊 Saved: holdout_experiment_policy.png")
    plt.show()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    holdout_experiment()
