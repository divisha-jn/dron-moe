import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):  
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class SimpleQNetwork(nn.Module):
    def __init__(self, state_size, num_actions, hidden_size=128):
        super(SimpleQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class BaselineDQNAgent:
    def __init__(self, state_size, num_actions, lr=0.001, gamma=0.95,
                 memory_size=10000, target_update_freq=1000, double_dqn=True):
        self.num_actions = num_actions
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.training_steps = 0

        self.model = SimpleQNetwork(state_size, num_actions)
        self.target_model = SimpleQNetwork(state_size, num_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_size)

        self.epsilon = 0.3
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

    def select_action(self, state, opponent_obs=None, training=True):
        # opponent_obs is accepted for API consistency with DRON-MoE but not used in baseline
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            action = q_values.argmax().item()
        self.model.train()  
        return action

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None

        batch = self.memory.sample(batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)  # Unpack `done`

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)  # Convert `done` to tensor

        self.model.train()
        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_q_online = self.model(next_states)
                next_actions = next_q_online.argmax(1)
                next_q_target = self.target_model(next_states)
                max_next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_model(next_states)
                max_next_q_values = next_q_values.max(1)[0]

            # Mask out terminal states so bootstrapping stops at episode end
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def remember(self, state, action, next_state, reward, done, opponent_obs=None):
        # opponent_obs is accepted for API consistency with DRON-MoE but not used in baseline
        self.memory.push(state, action, next_state, reward, done)  # Pass `done`

    def inspect(self, state):
        self.model.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            q = self.model(s).squeeze(0).cpu().numpy()
        self.model.train()  # Restore train mode after inspect
        return q
