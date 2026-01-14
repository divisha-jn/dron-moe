# dron_moe.py - DRON-MoE Implementation

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# =============================================================================
# PART 1: REPLAY MEMORY (The agent's diary)
# =============================================================================

class ReplayMemory:
    def __init__(self, capacity=10000):
        """Store experiences to learn from later"""
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, next_state, reward, opponent_obs):
        """Save one experience"""
        self.memory.append((state, action, next_state, reward, opponent_obs))
    
    def sample(self, batch_size):
        """Randomly grab some memories"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


# =============================================================================
# PART 2: OPPONENT NETWORK (Understands opponent behavior)
# =============================================================================

class OpponentNetwork(nn.Module):
    def __init__(self, opponent_obs_size, hidden_size=50):
        """Learn to represent opponent's strategy"""
        super(OpponentNetwork, self).__init__()
        self.fc1 = nn.Linear(opponent_obs_size, hidden_size)
        self.relu = nn.ReLU()
        
    def forward(self, opponent_obs):
        """Convert opponent observations into a summary"""
        h_o = self.relu(self.fc1(opponent_obs))
        return h_o

# =============================================================================
# PART 3: EXPERT NETWORKS
# =============================================================================

class ExpertNetwork(nn.Module):
    def __init__(self, state_size, num_actions, hidden_size=128):
        super(ExpertNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_actions)
        
    def forward(self, state_representation):
        x = self.relu(self.fc1(state_representation))
        q_values = self.fc2(x)
        return q_values


# =============================================================================
# PART 4: GATING NETWORK
# =============================================================================

class GatingNetwork(nn.Module):
    def __init__(self, opponent_hidden_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(opponent_hidden_size, num_experts)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, opponent_representation):
        logits = self.fc(opponent_representation)
        weights = self.softmax(logits)
        return weights
    

# =============================================================================
# PART 5: DRON-MoE (The Complete Model)
# =============================================================================

class DRON_MoE(nn.Module):
    def __init__(self, state_size, opponent_obs_size, num_actions, 
                 num_experts=3, hidden_size=128, opponent_hidden_size=50):
        """
        The complete DRON-MoE model
        
        state_size: size of game state (like 10)
        opponent_obs_size: size of opponent observations (like 5)
        num_actions: number of possible actions (like 5)
        num_experts: how many expert networks (default 3)
        hidden_size: neurons in expert networks (default 128)
        opponent_hidden_size: neurons in opponent network (default 50)
        """
        super(DRON_MoE, self).__init__()
        
        self.num_experts = num_experts
        self.num_actions = num_actions
        
        # 1. Opponent Network - understands the opponent
        self.opponent_net = OpponentNetwork(opponent_obs_size, opponent_hidden_size)
        
        # 2. State Network - processes the game state
        self.state_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU()
        )
        
        # 3. Multiple Expert Networks - one for each strategy
        self.experts = nn.ModuleList([
            ExpertNetwork(hidden_size, num_actions, hidden_size) 
            for _ in range(num_experts)
        ])
        
        # 4. Gating Network - decides which experts to trust
        self.gating_net = GatingNetwork(opponent_hidden_size, num_experts)
        
    def forward(self, state, opponent_obs):
        """
        The main forward pass - predict Q-values given state and opponent
        
        state: current game state (tensor)
        opponent_obs: observations about opponent (tensor)
        
        Returns: final Q-values (weighted combination from all experts)
        """
        # Step 1: Process the game state
        h_s = self.state_net(state)  # Shape: (batch_size, hidden_size)
        
        # Step 2: Process opponent observations
        h_o = self.opponent_net(opponent_obs)  # Shape: (batch_size, opponent_hidden_size)
        
        # Step 3: Get gating weights (which experts to trust)
        gating_weights = self.gating_net(h_o)  # Shape: (batch_size, num_experts)
        
        # Step 4: Get Q-values from each expert
        expert_q_values = []
        for expert in self.experts:
            q = expert(h_s)  # Each expert predicts Q-values
            expert_q_values.append(q.unsqueeze(2))  # Add dimension for stacking
        
        # Stack all expert predictions
        # Shape: (batch_size, num_actions, num_experts)
        expert_q_values = torch.cat(expert_q_values, dim=2)
        
        # Step 5: Combine expert predictions using gating weights
        # Expand gating weights for broadcasting
        gating_weights = gating_weights.unsqueeze(1)  # Shape: (batch_size, 1, num_experts)
        
        # Weighted sum: final Q-values
        final_q_values = torch.sum(expert_q_values * gating_weights, dim=2)
        # Shape: (batch_size, num_actions)
        
        return final_q_values


# =============================================================================
# PART 6: DRON-MoE AGENT (Training and Decision Making)
# =============================================================================

class DRON_MoE_Agent:
    def __init__(self, state_size, opponent_obs_size, num_actions, 
                 num_experts=3, lr=0.0005, gamma=0.9, memory_size=10000):
        """
        The agent that uses DRON-MoE to play the game
        
        lr: learning rate (how fast to learn, default 0.0005)
        gamma: discount factor (how much to value future rewards, default 0.9)
        """
        self.num_actions = num_actions
        self.gamma = gamma
        
        # Create the DRON-MoE model
        self.model = DRON_MoE(state_size, opponent_obs_size, num_actions, num_experts)
        
        # Optimizer for learning
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Replay memory
        self.memory = ReplayMemory(memory_size)
        
        # Exploration parameters
        self.epsilon = 0.3  # Start with 30% random exploration
        self.epsilon_min = 0.1  # Decay to 10% minimum
        self.epsilon_decay = 0.995  # Decay rate
        
    def select_action(self, state, opponent_obs, training=True):
        """
        Choose an action using epsilon-greedy strategy
        
        state: current game state
        opponent_obs: current opponent observations
        training: if True, use exploration; if False, always pick best action
        """
        # Exploration: sometimes pick random action (helps learning)
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Exploitation: pick best action based on Q-values
        with torch.no_grad():  # Don't track gradients for inference
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            opponent_tensor = torch.FloatTensor(opponent_obs).unsqueeze(0)
            
            q_values = self.model(state_tensor, opponent_tensor)
            action = q_values.argmax().item()  # Pick action with highest Q-value
            
        return action
    
    def train_step(self, batch_size=64):
        """
        Learn from past experiences
        
        batch_size: how many experiences to learn from at once
        """
        # Need enough memories to learn from
        if len(self.memory) < batch_size:
            return None
        
        # Sample random batch from memory
        batch = self.memory.sample(batch_size)
        
        # Unpack the batch
        states, actions, next_states, rewards, opponent_obs_list = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        opponent_obs_batch = torch.FloatTensor(opponent_obs_list)
        
        # Current Q-values
        current_q_values = self.model(states, opponent_obs_batch)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (what we want to learn towards)
        with torch.no_grad():
            next_q_values = self.model(next_states, opponent_obs_batch)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values
        
        # Calculate loss (how wrong our predictions are)
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Update the model
        self.optimizer.zero_grad()  # Clear old gradients
        loss.backward()  # Calculate new gradients
        self.optimizer.step()  # Update weights
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def remember(self, state, action, next_state, reward, opponent_obs):
        """Save experience to memory"""
        self.memory.push(state, action, next_state, reward, opponent_obs)

