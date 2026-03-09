# DRON-MoE Implementation

Deep Reinforcement Opponent Network with Mixture of Experts (DRON-MoE) for ad hoc teamwork in cooperative multi-agent settings. Compared against a Baseline Double DQN agent.

---

## File Structure

```
├── example_usage.py        # Environment (CooperativeReaching)
├── dron_moe.py             # DRON-MoE network and agent
├── baseline_dqn.py         # Baseline Double DQN agent
├── logger.py               # Probe logging and plots
└── comparison_experiment.py # Training, testing, and experiments
```

---

## Environment — `example_usage.py`

7×7 grid. Agent and opponent start at center. Opponent secretly picks a target corner based on its personality type and walks toward it. Agent must reach the same corner.

**Coordinates:** Standard math (x→right, y→up). Corners:
- `(0,6)` top-left, `(6,6)` top-right, `(0,0)` bottom-left, `(6,0)` bottom-right

**Actions:** `0`=up, `1`=down, `2`=left, `3`=right

**State (10 values):**
- Agent position (x, y)
- Opponent position (x, y)
- Distance to each of 4 corners
- Time elapsed (normalised)
- Is opponent still moving?

**Opponent observation (6 values):**
- Last 3 observed movement deltas (dx, dy) — no internal goal info leaked

**Rewards:**
- Both reach same corner → `+10 + speed bonus`
- Agent reaches wrong corner → `-10`
- Timeout → `-5`
- Each step → `-0.1`

**Opponent types:**

| Type | Behaviour |
|---|---|
| `top_left_preference` | 70% chance targets top-left |
| `top_right_preference` | 70% chance targets top-right |
| `bottom_left_preference` | 70% chance targets bottom-left |
| `bottom_right_preference` | 70% chance targets bottom-right |
| `left_preference` | 40% each for top-left / bottom-left |
| `right_preference` | 40% each for top-right / bottom-right |
| `random` | 25% each corner |

**Seen vs Unseen:**
- Seen (trained on): `top_left`, `bottom_left`, `left_preference`
- Unseen (generalisation test): `top_right`, `bottom_right`, `right_preference`

---

## Pseudocode

### Replay Memory
```
CLASS ReplayMemory(capacity):
    memory = empty queue with max size capacity

    FUNCTION push(state, action, next_state, reward, done, ...):
        add (state, action, next_state, reward, done, ...) to memory

    FUNCTION sample(batch_size):
        return batch_size random items from memory
```

### Baseline DQN Agent — `baseline_dqn.py`
```
CLASS SimpleQNetwork(state_size, num_actions):
    layers: Linear → ReLU → Linear → ReLU → Linear
    INPUT:  state vector
    OUTPUT: Q-value for every action

CLASS BaselineDQNAgent:

    INIT:
        online_network  = SimpleQNetwork
        target_network  = copy of online_network (frozen)
        optimizer       = Adam
        memory          = ReplayMemory
        epsilon         = 0.3

    FUNCTION select_action(state):
        if random() < epsilon:
            return random action                                    // explore
        else:
            return action with highest Q from online_network(state) // exploit

    FUNCTION train_step():
        if memory.size() < 64: return

        batch = memory.sample(64)

        predicted_Q = online_network(state)[action_taken]

        // Double DQN target
        best_next_action = argmax( online_network(next_state) )
        target_Q = reward + gamma * target_network(next_state)[best_next_action] * (1 - done)

        loss = MSE(predicted_Q, target_Q)
        backpropagate loss through online_network

        every 1000 steps: copy online_network weights → target_network
        decay epsilon
```

### DRON-MoE Agent — `dron_moe.py`
```
CLASS DRON_MoE(state_size, opponent_obs_size, num_actions, num_experts=3):

    INIT:
        state_net    = Linear → ReLU          // compress state
        opponent_net = Linear → ReLU          // compress opponent obs
        experts      = [Linear → ReLU → Linear] × K   // K specialist Q-networks
        gating_net   = Linear → Softmax        // outputs K weights summing to 1

    FUNCTION forward(state, opponent_obs):
        h_state    = state_net(state)
        h_opponent = opponent_net(opponent_obs)

        weights      = gating_net(h_opponent)            // (K weights)
        q_per_expert = [expert(h_state) for each expert] // K × (A Q-values)

        final_Q = weighted sum of all expert Q-values    // blend by gating weights
        return final_Q

CLASS DRON_MoE_Agent:

    INIT:
        online_model  = DRON_MoE
        target_model  = copy of online_model (frozen)
        optimizer     = Adam
        memory        = ReplayMemory
        epsilon       = 0.3

    FUNCTION select_action(state, opponent_obs):
        if random() < epsilon:
            return random action
        else:
            return action with highest Q from online_model(state, opponent_obs)

    FUNCTION train_step():
        if memory.size() < 64: return

        batch = memory.sample(64)

        predicted_Q = online_model(state, opponent_obs)[action_taken]

        // Double DQN target
        best_next_action = argmax( online_model(next_state, next_opponent_obs) )
        target_Q = reward + gamma
                   * target_model(next_state, next_opponent_obs)[best_next_action]
                   * (1 - done)

        loss = MSE(predicted_Q, target_Q)
        backpropagate through state_net, opponent_net, experts, gating_net

        every 1000 steps: copy online_model weights → target_model
        decay epsilon
```

### Logger — `logger.py`
```
FUNCTION build_probes(opponent_types):
    for each opponent type and seed in:[1][2][3]
        reset env with fixed seed, play 5 fixed actions[4]
        save resulting (state, opponent_obs) as a probe

FUNCTION evaluate_probes(dron, baseline, probes, episode):
    for each probe in seen and unseen sets:
        ask dron:     get final Q-values, per-expert Q-values, gating weights
        ask baseline: get Q-values
        write both to CSV

FUNCTION plot_probe_trends(csv):
    for each episode checkpoint:
        average q_max, q_gap, gating weights across all probes
    plot:
        max Q over training        // agent confidence
        top-2 Q gap over training  // agent decisiveness
        gating weights over time   // expert specialisation
```

### Training Loop — `comparison_experiment.py`
```
FUNCTION train_agent(agent, opponent_types, num_episodes):
    for each episode:
        pick random opponent type → reset environment
        loop until done:
            agent picks action (epsilon-greedy)
            environment steps forward
            store experience in memory
            sample 64 random experiences → update network
        every 200 episodes: log probes to CSV

FUNCTION holdout_experiment():
    train both agents on SEEN opponents only
    test on SEEN   → measure in-distribution performance
    test on UNSEEN → measure generalisation
    compare generalisation gap: DRON-MoE vs Baseline
```

---

## Running Experiments

```bash
cd dron_q/
python comparison_experiment.py
```

```
Choose experiment:
1. Full comparison (all opponents)
2. Holdout experiment (generalisation test)
3. Both
```

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Grid size | 7×7 |
| Max steps per episode | 30 |
| Num experts (DRON-MoE) | 3 |
| Hidden size | 128 |
| Opponent hidden size | 50 |
| Learning rate | 0.001 |
| Gamma (discount) | 0.95 |
| Epsilon (start → min) | 0.3 → 0.1 |
| Epsilon decay | 0.995 per step |
| Replay buffer size | 10,000 |
| Batch size | 64 |
| Target update freq | every 1000 steps |
| Training episodes | 1000 (comparison) / 5000 (probe logging) |
| Success threshold | reward > 8 |
