# baseline_dqn.py - Standard DQN without opponent modeling

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# =============================================================================
# REPLAY MEMORY (Same as DRON-MoE)
# =============================================================================

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, next_state, reward):
        """Note: NO opponent_obs stored - baseline ignores opponent info"""
        self.memory.append((state, action, next_state, reward))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


# =============================================================================
# SIMPLE Q-NETWORK (No experts, no gating, no opponent network)
# =============================================================================

class SimpleQNetwork(nn.Module):
    def __init__(self, state_size, num_actions, hidden_size=128):
        """
        Basic DQN: Just state → hidden layers → Q-values
        No opponent modeling at all!
        """
        super(SimpleQNetwork, self).__init__()
        
        # Two hidden layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)
        self.relu = nn.ReLU()
        
    def forward(self, state):
        """
        Simple forward pass
        
        Input: state only (ignores opponent observations)
        Output: Q-values for each action
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


# =============================================================================
# BASELINE DQN AGENT
# =============================================================================

class BaselineDQNAgent:
    def __init__(self, state_size, num_actions, lr=0.001, gamma=0.95, memory_size=10000):
        """
        Standard DQN agent - treats environment as stationary
        Ignores opponent observations completely
        """
        self.num_actions = num_actions
        self.gamma = gamma
        
        # Create simple Q-network
        self.model = SimpleQNetwork(state_size, num_actions)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Replay memory
        self.memory = ReplayMemory(memory_size)
        
        # Exploration parameters
        self.epsilon = 0.3
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
    def select_action(self, state, opponent_obs=None, training=True):
        """
        Select action - NOTE: opponent_obs is ignored!
        
        Baseline DQN doesn't use opponent information at all
        """
        # Exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)  # Only uses state!
            action = q_values.argmax().item()
            
        return action
    
    def train_step(self, batch_size=64):
        """Standard DQN training with temporal difference learning"""
        if len(self.memory) < batch_size:
            return None
        
        # Sample batch
        batch = self.memory.sample(batch_size)
        states, actions, next_states, rewards = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        
        # Current Q-values
        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (Bellman equation)
        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values
        
        # Calculate loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def remember(self, state, action, next_state, reward, opponent_obs=None):
        """
        Store experience - opponent_obs ignored
        """
        self.memory.push(state, action, next_state, reward)
