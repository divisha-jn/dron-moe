# DRON-MoE Policy Implementation

Policy-based (PPO) implementation of DRON-MoE vs Baseline Policy agent for ad hoc teamwork in cooperative multi-agent settings. Uses the same `CooperativeReaching` environment as the Q-value implementation.

---

## File Structure

```
policy/
├── example_usage.py                  # Environment (CooperativeReaching)
├── baseline_policy.py                # Baseline PPO agent (ActorCritic)
├── dron_moe_policy.py                # DRON-MoE PPO agent
├── logger_policy.py                  # Probe logging and plots
└── comparison_experiment_policy.py   # Training, testing, and experiments
```

---

## Environment — `example_usage.py`

Identical to Q-value implementation. See `q_value/README.md` for full environment details.

**Key values for policy agents:**
- `state_size = 10`
- `opponent_obs_size = 6` (last 3 movement deltas as dx, dy)
- `num_actions = 4` (up, down, left, right)

---

## Pseudocode

### Episode Buffer — shared by both agents
```
CLASS EpisodeBuffer:
    stores per-step: state, action, reward, value, log_prob, done, opponent_obs
    also stores:     last_next_state, last_next_opp_obs  ← for bootstrap

    FUNCTION push(state, action, reward, value, log_prob, done, opponent_obs):
        append all fields

    FUNCTION get():
        return all fields as numpy arrays

    FUNCTION clear():
        reset all lists and bootstrap fields
```

### Baseline PPO Agent — `baseline_policy.py`
```
CLASS ActorCriticNetwork(state_size, opponent_obs_size, num_actions):
    shared trunk: Linear → ReLU → Linear → ReLU
    actor head:   Linear → action logits
    critic head:  Linear → single value

    INPUT:  [state + opponent_obs] concatenated
    OUTPUT: action logits, value estimate

CLASS BaselinePolicyAgent:

    INIT:
        model     = ActorCriticNetwork
        optimizer = Adam
        scheduler = StepLR (decay LR every 1000 steps)
        buffer    = EpisodeBuffer
        running reward normalisation stats

    FUNCTION select_action(state, opponent_obs):
        forward pass through network
        if training: sample from Categorical(logits)  ← stochastic
        if testing:  take argmax                       ← deterministic
        store (state, action, value, log_prob, opponent_obs) for remember()

    FUNCTION remember(reward, done, next_state, next_opponent_obs):
        push stored transition into buffer
        update buffer.last_next_state for bootstrap

    FUNCTION train_step():
        called once per episode — trains on full trajectory then clears buffer

        1. normalise rewards using running mean/std
        2. bootstrap next_value from last_next_state (0 if episode ended)
        3. compute GAE advantages and returns
        4. normalise advantages
        5. run PPO update for `epochs` passes:
              recompute log_probs under current policy
              ratio = exp(new_log_prob - old_log_prob)
              actor_loss  = -min(ratio × A, clip(ratio, 1±ε) × A)
              critic_loss = MSE(predicted_value, returns)
              entropy     = mean policy entropy
              loss = actor_loss + 0.5 × critic_loss - 0.02 × entropy
              clip gradients, backpropagate
        6. clear buffer
```

### GAE — used by both agents
```
FUNCTION compute_gae(rewards, values, dones, next_value, γ=0.99, λ=0.95):
    append next_value to values
    gae = 0
    for t from last step to first:
        if done[t]:
            delta = reward[t] - value[t]   ← episode ended, no next state
            gae   = delta
        else:
            delta = reward[t] + γ × value[t+1] - value[t]
            gae   = delta + γ × λ × gae    ← blend TD and MC
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns
```

### DRON-MoE PPO Agent — `dron_moe_policy.py`
```
CLASS ExpertActorCritic(state_size, num_actions):
    shared trunk: Linear → ReLU
    actor head:   Linear → action logits
    critic head:  Linear → value
    each expert has its own opinion on both what to do AND how good the state is

CLASS DRON_MoE_Policy(state_size, opponent_obs_size, num_actions, num_experts=3):

    INIT:
        state_net    = Linear → ReLU          // compress state
        opponent_net = Linear → ReLU          // compress opponent obs
        experts      = [ExpertActorCritic] × K
        gating_net   = Linear → Softmax       // K weights from opponent representation

    FUNCTION forward(state, opponent_obs):
        h_state    = state_net(state)
        h_opponent = opponent_net(opponent_obs)

        weights    = gating_net(h_opponent)               // (K weights)

        for each expert:
            get expert_logits (A,) and expert_value (1,)

        combined_logits = weighted sum of expert logits   // (A,)
        combined_value  = weighted sum of expert values   // critic loss does NOT
                          using detached weights          // flow into gating network

        return combined_logits, combined_value

CLASS DRON_MoE_PolicyAgent:
    identical PPO training loop to BaselinePolicyAgent
    only difference: calls DRON_MoE_Policy instead of ActorCriticNetwork
    entropy_coef = 0.05 (vs 0.02 for baseline — pushed harder to stay diverse)
```

### Logger — `logger_policy.py`
```
FUNCTION build_probes(opponent_types):
    for each opponent type and seed in:[1][2][3]
        reset env with fixed seed, play 5 fixed actions[4]
        save resulting (state, opponent_obs) as a probe

FUNCTION evaluate_probes(dron, baseline, probes, episode):
    for each probe in seen and unseen sets:
        ask dron:     get action_probs, value, gating_weights → log to CSV
        ask baseline: get action_probs, value               → log to CSV
    logged fields per probe:
        chosen_action, entropy, value, p_a0..p_a3, w_k0..w_kK

FUNCTION plot_probe_trends(csv):
    for each episode checkpoint:
        average entropy and gating weights across all probes
    plot:
        entropy over training    // is policy becoming more decisive?
        gating weights over time // are experts specialising?
```

### Training Loop — `comparison_experiment_policy.py`
```
FUNCTION train_agent(agent, opponent_types, num_episodes):
    for each episode:
        pick random opponent type → reset environment
        loop until done:
            agent selects action (stochastic policy sampling)
            environment steps forward
            agent.remember(reward, done, next_state, next_opponent_obs)
        agent.train_step()  ← train on full episode trajectory, then clear buffer
        every 200 episodes: evaluate_probes → log to CSV

FUNCTION test_agent(agent, opponent_types):
    for each opponent type:
        run 50 episodes with greedy policy (training=False)
        record success rate and average reward

FUNCTION holdout_experiment():
    train both agents on SEEN opponents only
    test on SEEN   → in-distribution performance
    test on UNSEEN → generalisation
    compare generalisation gap: DRON-MoE-Policy vs Baseline Policy
    plot 6-panel figure: rewards, success, actor loss, critic loss, seen bars, unseen bars ⭐
```

---

## Key Differences from Q-Value Implementation

| | Q-Value (DQN) | Policy (PPO) |
|---|---|---|
| What it learns | Q(s,a) for each action | π(a\|s) + V(s) |
| Exploration | Epsilon-greedy | Stochastic sampling from policy |
| Memory | Random replay buffer | Episode buffer — cleared after each update |
| Training trigger | Every step | End of episode |
| Loss function | Bellman / TD error | PPO clipped surrogate + value + entropy |
| Extra metrics | Q-max, Q-gap | Actor loss, critic loss, entropy |
| Probe logging | Q-values, gating weights | Action probabilities, entropy, gating weights |

---

## Running

```bash
cd dron_policy/
python comparison_experiment_policy.py
```

---

## Key Hyperparameters

| Parameter | DRON-MoE-Policy | Baseline Policy |
|---|---|---|
| State size | 10 | 10 |
| Opponent obs size | 6 | 6 |
| Num experts | 3 | — |
| Hidden size | 128 | 128 |
| Opponent hidden size | 50 | — |
| Learning rate | 0.00005 | 0.00005 |
| Gamma (discount) | 0.99 | 0.99 |
| GAE lambda | 0.95 | 0.95 |
| Epsilon clip | 0.2 | 0.2 |
| PPO epochs per update | 4 | 4 |
| Value coefficient | 0.5 | 0.5 |
| Entropy coefficient | 0.05 | 0.02 |
| LR decay | every 1000 steps × 0.95 | every 1000 steps × 0.95 |
| Grad clip | 0.5 | 0.5 |
| Training episodes | 5000 | 5000 |
| Test episodes per type | 50 | 50 |
| Success threshold | reward > 8 | reward > 8 |
| Probe log frequency | every 200 episodes | every 200 episodes |
