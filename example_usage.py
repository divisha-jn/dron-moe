# example_usage.py - Complete DRON-MoE Training on Cooperative Reaching

import numpy as np
import torch
from dron_moe import DRON_MoE_Agent

# =============================================================================
# COOPERATIVE REACHING ENVIRONMENT
# =============================================================================

class CooperativeReaching:
    """
    Cooperative Reaching Game
    
    - 5x5 grid with 4 corners (goals)
    - Agent and opponent both start in center
    - Goal: Reach the SAME corner as the opponent
    - Challenge: You don't know which corner opponent will pick!
    - Opponent types have different preferences
    
    Corners:
      0 (top-left)     1 (top-right)
      
      2 (bottom-left)  3 (bottom-right)
    """
    def __init__(self, opponent_type='top_left_preference'):
        self.grid_size = 5
        self.opponent_type = opponent_type
        
        # Define corner positions
        self.corners = {
            0: (0, 0),      # Top-left
            1: (4, 0),      # Top-right
            2: (0, 4),      # Bottom-left
            3: (4, 4)       # Bottom-right
        }
        
        # Opponent preferences (probabilities for each corner)
        self.opponent_behaviors = {
            'top_left_preference': [0.7, 0.1, 0.1, 0.1],      # 70% goes top-left
            'top_right_preference': [0.1, 0.7, 0.1, 0.1],     # 70% goes top-right
            'bottom_left_preference': [0.1, 0.1, 0.7, 0.1],   # 70% goes bottom-left
            'bottom_right_preference': [0.1, 0.1, 0.1, 0.7],  # 70% goes bottom-right
            'left_preference': [0.4, 0.1, 0.4, 0.1],          # Prefers left side
            'right_preference': [0.1, 0.4, 0.1, 0.4],         # Prefers right side
            'random': [0.25, 0.25, 0.25, 0.25]                # Completely random
        }
        
        self.reset()
    
    def reset(self):
        """Start a new episode"""
        # Both start in center
        self.agent_x, self.agent_y = 2, 2
        self.opponent_x, self.opponent_y = 2, 2
        
        # Opponent picks their target corner based on their preference
        probs = self.opponent_behaviors[self.opponent_type]
        self.opponent_target = np.random.choice([0, 1, 2, 3], p=probs)
        self.opponent_goal_x, self.opponent_goal_y = self.corners[self.opponent_target]
        
        self.agent_reached = False
        self.opponent_reached = False
        self.agent_corner = None
        self.steps = 0
        self.max_steps = 20  # Enough time to reach any corner
        
        return self.get_state(), self.get_opponent_obs()
    
    def get_state(self):
        """
        State representation:
        [agent_x, agent_y, opponent_x, opponent_y,
         dist_to_corner0, dist_to_corner1, dist_to_corner2, dist_to_corner3,
         steps_remaining, opponent_is_moving]
        """
        # Calculate distances to all corners
        distances = []
        for corner_pos in self.corners.values():
            dist = abs(self.agent_x - corner_pos[0]) + abs(self.agent_y - corner_pos[1])
            distances.append(dist / 8.0)  # Normalize (max distance is 8)
        
        # Opponent movement indicator
        opponent_moving = 1.0 if not self.opponent_reached else 0.0
        
        state = [
            self.agent_x / 4.0,
            self.agent_y / 4.0,
            self.opponent_x / 4.0,
            self.opponent_y / 4.0,
            distances[0],
            distances[1],
            distances[2],
            distances[3],
            self.steps / self.max_steps,
            opponent_moving
        ]
        
        return np.array(state, dtype=np.float32)
    
    def get_opponent_obs(self):
        """
        Opponent observations (features that hint at their strategy):
        [moves_toward_left, moves_toward_right, moves_toward_top, 
         moves_toward_bottom, consistency_score]
        """
        target_x, target_y = self.corners[self.opponent_target]
        
        # These features encode opponent's target preference
        moves_left = 1.0 if target_x == 0 else 0.0
        moves_right = 1.0 if target_x == 4 else 0.0
        moves_top = 1.0 if target_y == 0 else 0.0
        moves_bottom = 1.0 if target_y == 4 else 0.0
        
        # Consistency (how predictable is this opponent?)
        probs = self.opponent_behaviors[self.opponent_type]
        consistency = max(probs)  # Higher = more predictable
        
        obs = [moves_left, moves_right, moves_top, moves_bottom, consistency]
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, agent_action):
        """
        Execute one step
        
        Actions: 0=up, 1=down, 2=left, 3=right
        
        Returns: next_state, opponent_obs, reward, done
        """
        self.steps += 1
        
        # Move agent
        if agent_action == 0:  # up
            self.agent_y = max(0, self.agent_y - 1)
        elif agent_action == 1:  # down
            self.agent_y = min(4, self.agent_y + 1)
        elif agent_action == 2:  # left
            self.agent_x = max(0, self.agent_x - 1)
        elif agent_action == 3:  # right
            self.agent_x = min(4, self.agent_x + 1)
        
        # Check if agent reached a corner
        agent_corner = self._get_corner(self.agent_x, self.agent_y)
        if agent_corner is not None:
            self.agent_reached = True
            self.agent_corner = agent_corner
        
        # Move opponent toward their target corner
        if not self.opponent_reached:
            self._move_opponent()
        
        # Check if opponent reached their target corner
        if (self.opponent_x, self.opponent_y) == (self.opponent_goal_x, self.opponent_goal_y):
            self.opponent_reached = True
        
        # Calculate reward
        reward, done = self._calculate_reward()
        
        return self.get_state(), self.get_opponent_obs(), reward, done
    
    def _get_corner(self, x, y):
        """Check if position is a corner, return corner ID or None"""
        for corner_id, (cx, cy) in self.corners.items():
            if x == cx and y == cy:
                return corner_id
        return None
    
    def _move_opponent(self):
        """Move opponent one step toward their target corner"""
        # Move horizontally first
        if self.opponent_x < self.opponent_goal_x:
            self.opponent_x += 1
        elif self.opponent_x > self.opponent_goal_x:
            self.opponent_x -= 1
        # Then move vertically
        elif self.opponent_y < self.opponent_goal_y:
            self.opponent_y += 1
        elif self.opponent_y > self.opponent_goal_y:
            self.opponent_y -= 1
    
    def _calculate_reward(self):
        """Calculate reward based on coordination success"""
        done = False
        reward = 0.0
        
        # Both reached corners
        if self.agent_reached and self.opponent_reached:
            done = True
            
            # SUCCESS: Same corner!
            if self.agent_corner == self.opponent_target:
                reward = 10.0
                # Bonus for finishing faster
                time_bonus = (self.max_steps - self.steps) * 0.5
                reward += time_bonus
            
            # FAILURE: Different corners
            else:
                reward = -10.0
        
        # Timeout
        elif self.steps >= self.max_steps:
            done = True
            reward = -5.0  # Penalty for not coordinating in time
        
        # Still playing: small step penalty
        else:
            reward = -0.1
            
            # Small reward for getting closer to opponent's target
            if not self.agent_reached:
                dist_to_opp_target = abs(self.agent_x - self.opponent_goal_x) + \
                                    abs(self.agent_y - self.opponent_goal_y)
                
                # Note: We're "cheating" a bit here for training purposes
                # In real coordination, agent wouldn't know opponent's target
                # But the MoE should learn patterns from opponent_obs
                if dist_to_opp_target < 2:
                    reward += 0.2
        
        return reward, done
    
    def render(self):
        """Visualize the current state (optional, for debugging)"""
        grid = [['.' for _ in range(5)] for _ in range(5)]
        
        # Mark corners
        for corner_id, (cx, cy) in self.corners.items():
            grid[cy][cx] = str(corner_id)
        
        # Mark opponent
        if not self.opponent_reached:
            grid[self.opponent_y][self.opponent_x] = 'O'
        else:
            grid[self.opponent_goal_y][self.opponent_goal_x] = 'Ø'
        
        # Mark agent
        if not self.agent_reached:
            grid[self.agent_y][self.agent_x] = 'A'
        else:
            grid[self.agent_y][self.agent_x] = 'Å'
        
        print("\n" + "-" * 11)
        for row in grid:
            print("| " + " ".join(row) + " |")
        print("-" * 11)
        print(f"Opponent target: Corner {self.opponent_target}")


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_dron_moe_cooperative():
    """Train DRON-MoE agent on cooperative reaching"""
    
    # Mix of different opponent types during training
    opponent_types = [
        'top_left_preference',
        'top_right_preference', 
        'bottom_left_preference',
        'bottom_right_preference',
        'left_preference',
        'right_preference'
    ]
    
    # Create DRON-MoE agent
    agent = DRON_MoE_Agent(
        state_size=10,           # 10 state features
        opponent_obs_size=5,     # 5 opponent observation features
        num_actions=4,           # 4 directions (no "stay" action)
        num_experts=3,           # 3 experts to learn different strategies
        lr=0.001,               # Slightly higher learning rate
        gamma=0.95,             # High discount (future coordination matters)
        memory_size=10000
    )
    
    # Training parameters
    num_episodes = 1000
    batch_size = 64
    
    # Track performance
    episode_rewards = []
    success_rate = []
    
    print("Starting COOPERATIVE REACHING training...")
    print("Agent must learn to predict opponent's corner choice!")
    print("-" * 60)
    
    for episode in range(num_episodes):
        # Randomly pick opponent type for this episode
        opponent_type = np.random.choice(opponent_types)
        env = CooperativeReaching(opponent_type=opponent_type)
        
        # Reset environment
        state, opponent_obs = env.reset()
        episode_reward = 0
        done = False
        
        # Play one episode
        while not done:
            # Select action
            action = agent.select_action(state, opponent_obs, training=True)
            
            # Execute action
            next_state, next_opponent_obs, reward, done = env.step(action)
            
            # Store experience
            agent.remember(state, action, next_state, reward, opponent_obs)
            
            # Train
            loss = agent.train_step(batch_size)
            
            # Update
            state = next_state
            opponent_obs = next_opponent_obs
            episode_reward += reward
        
        # Track performance
        episode_rewards.append(episode_reward)
        success = 1 if episode_reward > 5 else 0  # Success if reward > 5
        success_rate.append(success)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_success = np.mean(success_rate[-100:]) * 100
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Success Rate: {avg_success:.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print("-" * 60)
    print("Training complete!")
    print(f"Final success rate (last 100 episodes): {np.mean(success_rate[-100:]) * 100:.1f}%")
    
    return agent, episode_rewards


# =============================================================================
# TESTING FUNCTION
# =============================================================================

def test_cooperative_agent(agent, num_episodes=20, opponent_type='top_left_preference'):
    """Test the trained agent on cooperative reaching"""
    
    env = CooperativeReaching(opponent_type=opponent_type)
    successes = 0
    total_rewards = []
    
    print(f"\nTesting against '{opponent_type}' opponent...")
    print("-" * 60)
    
    for episode in range(num_episodes):
        state, opponent_obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            action = agent.select_action(state, opponent_obs, training=False)
            state, opponent_obs, reward, done = env.step(action)
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
        
        if episode_reward > 5:  # Successful coordination
            successes += 1
            result = "✓ COORDINATED"
        else:
            result = "✗ FAILED"
        
        print(f"Episode {episode + 1}: {result} | "
              f"Reward: {episode_reward:.2f} | Steps: {steps} | "
              f"Agent→Corner{env.agent_corner if env.agent_reached else '?'} "
              f"Opponent→Corner{env.opponent_target}")
    
    success_rate = successes / num_episodes * 100
    avg_reward = np.mean(total_rewards)
    
    print("-" * 60)
    print(f"Success Rate: {success_rate:.1f}% | Average Reward: {avg_reward:.2f}")
    
    return success_rate, avg_reward


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Train the agent
    print("=" * 60)
    print("TRAINING DRON-MoE ON COOPERATIVE REACHING")
    print("=" * 60)
    
    trained_agent, rewards = train_dron_moe_cooperative()
    
    # Test against each opponent type
    print("\n" + "=" * 60)
    print("TESTING PHASE - Can the agent generalize?")
    print("=" * 60)
    
    test_results = {}
    for opp_type in ['top_left_preference', 'top_right_preference', 
                     'bottom_left_preference', 'bottom_right_preference',
                     'left_preference', 'right_preference', 'random']:
        success_rate, avg_reward = test_cooperative_agent(
            trained_agent, num_episodes=20, opponent_type=opp_type
        )
        test_results[opp_type] = success_rate
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for opp_type, success in test_results.items():
        print(f"{opp_type:25s}: {success:5.1f}% success rate")
    print(f"\nOverall Average: {np.mean(list(test_results.values())):.1f}%")
