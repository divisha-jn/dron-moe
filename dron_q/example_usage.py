import numpy as np
from collections import deque


# =============================================================================
# OPPONENT SETS
# =============================================================================

SEEN_OPPONENTS   = ['top_left_preference', 'bottom_left_preference', 'left_preference']
UNSEEN_OPPONENTS = ['top_right_preference', 'bottom_right_preference', 'right_preference']
ALL_TRAIN_OPPONENTS = SEEN_OPPONENTS


# =============================================================================
# ENVIRONMENT
# =============================================================================

class CooperativeReaching:
    def __init__(self, opponent_type='top_left_preference'):
        self.grid_size = 7
        self.opponent_type = opponent_type

        # Standard math coordinates: x goes right, y goes up
        self.corners = {
            0: (0, 6),   # top-left
            1: (6, 6),   # top-right
            2: (0, 0),   # bottom-left
            3: (6, 0)    # bottom-right
        }

        self.opponent_behaviors = {
            'top_left_preference':    [0.7, 0.1, 0.1, 0.1],
            'top_right_preference':   [0.1, 0.7, 0.1, 0.1],
            'bottom_left_preference': [0.1, 0.1, 0.7, 0.1],
            'bottom_right_preference':[0.1, 0.1, 0.1, 0.7],
            'left_preference':        [0.4, 0.1, 0.4, 0.1],
            'right_preference':       [0.1, 0.4, 0.1, 0.4],
            'random':                 [0.25, 0.25, 0.25, 0.25]
        }

        self.reset()

    def reset(self):
        self.agent_x, self.agent_y = 3, 3
        self.opponent_x, self.opponent_y = 3, 3

        probs = self.opponent_behaviors[self.opponent_type]
        self.opponent_target = np.random.choice([0, 1, 2, 3], p=probs)
        self.opponent_goal_x, self.opponent_goal_y = self.corners[self.opponent_target]

        self.agent_reached = False
        self.opponent_reached = False
        self.agent_corner = None
        self.steps = 0
        self.max_steps = 30

        # Track last 3 opponent moves as (dx, dy) — fully observable, no goal leakage
        self.opponent_move_history = deque([(0, 0)] * 3, maxlen=3)
        self.prev_opponent_x = self.opponent_x
        self.prev_opponent_y = self.opponent_y

        return self.get_state(), self.get_opponent_obs()

    def get_state(self):
        distances = []
        for corner_pos in self.corners.values():
            dist = abs(self.agent_x - corner_pos[0]) + abs(self.agent_y - corner_pos[1])
            distances.append(dist / 12.0)

        opponent_moving = 1.0 if not self.opponent_reached else 0.0

        state = [
            self.agent_x / 6.0,
            self.agent_y / 6.0,
            self.opponent_x / 6.0,
            self.opponent_y / 6.0,
            distances[0], distances[1], distances[2], distances[3],
            self.steps / self.max_steps,
            opponent_moving
        ]
        return np.array(state, dtype=np.float32)

    def get_opponent_obs(self):
        # Last 3 observed movement deltas — shape (6,)
        obs = []
        for dx, dy in self.opponent_move_history:
            obs.append(dx / 6.0)
            obs.append(dy / 6.0)
        return np.array(obs, dtype=np.float32)

    def step(self, agent_action):
        self.steps += 1

        # Actions: 0=up, 1=down, 2=left, 3=right (standard math coords)
        if agent_action == 0:
            self.agent_y = min(6, self.agent_y + 1)   # up
        elif agent_action == 1:
            self.agent_y = max(0, self.agent_y - 1)   # down
        elif agent_action == 2:
            self.agent_x = max(0, self.agent_x - 1)   # left
        elif agent_action == 3:
            self.agent_x = min(6, self.agent_x + 1)   # right

        agent_corner = self._get_corner(self.agent_x, self.agent_y)
        if agent_corner is not None:
            self.agent_reached = True
            self.agent_corner = agent_corner

        if not self.opponent_reached:
            self.prev_opponent_x = self.opponent_x
            self.prev_opponent_y = self.opponent_y
            self._move_opponent()
            dx = self.opponent_x - self.prev_opponent_x
            dy = self.opponent_y - self.prev_opponent_y
            self.opponent_move_history.append((dx, dy))

        if (self.opponent_x, self.opponent_y) == (self.opponent_goal_x, self.opponent_goal_y):
            self.opponent_reached = True

        reward, done = self._calculate_reward()
        return self.get_state(), self.get_opponent_obs(), reward, done

    def _get_corner(self, x, y):
        for corner_id, (cx, cy) in self.corners.items():
            if x == cx and y == cy:
                return corner_id
        return None

    def _move_opponent(self):
        # L-shaped path: correct x first, then y
        if self.opponent_x < self.opponent_goal_x:
            self.opponent_x += 1
        elif self.opponent_x > self.opponent_goal_x:
            self.opponent_x -= 1
        elif self.opponent_y < self.opponent_goal_y:
            self.opponent_y += 1
        elif self.opponent_y > self.opponent_goal_y:
            self.opponent_y -= 1

    def _calculate_reward(self):
        done = False
        reward = 0.0

        if self.agent_reached and self.opponent_reached:
            done = True
            if self.agent_corner == self.opponent_target:
                reward = 10.0 + (self.max_steps - self.steps) * 0.5
            else:
                reward = -10.0
        elif self.steps >= self.max_steps:
            done = True
            reward = -5.0
        else:
            reward = -0.1

        return reward, done
