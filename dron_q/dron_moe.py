import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done, opponent_obs, next_opponent_obs):  
        self.memory.append((state, action, next_state, reward, done, opponent_obs, next_opponent_obs))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class OpponentNetwork(nn.Module):
    def __init__(self, opponent_obs_size, hidden_size=50):
        super(OpponentNetwork, self).__init__()
        self.fc1 = nn.Linear(opponent_obs_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, opponent_obs):
        return self.relu(self.fc1(opponent_obs))


class ExpertNetwork(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size=128):  
        super(ExpertNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state_representation):
        x = self.relu(self.fc1(state_representation))
        return self.fc2(x)


class GatingNetwork(nn.Module):
    def __init__(self, opponent_hidden_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(opponent_hidden_size, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, opponent_representation):
        logits = self.fc(opponent_representation)
        return self.softmax(logits)


class DRON_MoE(nn.Module):
    def __init__(self, state_size, opponent_obs_size, num_actions,
                 num_experts=3, hidden_size=128, opponent_hidden_size=50):
        super(DRON_MoE, self).__init__()

        self.num_experts = num_experts
        self.num_actions = num_actions

        self.opponent_net = OpponentNetwork(opponent_obs_size, opponent_hidden_size)

        self.state_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU()
        )

        self.experts = nn.ModuleList([
            ExpertNetwork(hidden_size, num_actions, hidden_size)  # input_size = hidden_size (output of state_net)
            for _ in range(num_experts)
        ])

        self.gating_net = GatingNetwork(opponent_hidden_size, num_experts)

    def forward(self, state, opponent_obs, return_expert_values=False):
        h_s = self.state_net(state)
        h_o = self.opponent_net(opponent_obs)
        gating_weights = self.gating_net(h_o)               # (B, K)

        expert_q_values = []
        for expert in self.experts:
            q = expert(h_s)                                  # (B, A)
            expert_q_values.append(q.unsqueeze(2))           # (B, A, 1)

        expert_q_values = torch.cat(expert_q_values, dim=2)  # (B, A, K)

        gating_expanded = gating_weights.unsqueeze(1)        # (B, 1, K)
        final_q_values = torch.sum(expert_q_values * gating_expanded, dim=2)  # (B, A)

        if return_expert_values:
            return final_q_values, expert_q_values, gating_weights

        return final_q_values


class DRON_MoE_Agent:
    def __init__(self, state_size, opponent_obs_size, num_actions,
                 num_experts=3, lr=0.0005, gamma=0.9, memory_size=10000,
                 target_update_freq=1000, double_dqn=True):

        self.num_actions = num_actions
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.training_steps = 0

        self.model = DRON_MoE(state_size, opponent_obs_size, num_actions, num_experts)
        self.target_model = DRON_MoE(state_size, opponent_obs_size, num_actions, num_experts)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_size)

        self.epsilon = 0.3
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

    def select_action(self, state, opponent_obs, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            opponent_tensor = torch.FloatTensor(opponent_obs).unsqueeze(0)
            q_values = self.model(state_tensor, opponent_tensor)
            action = q_values.argmax().item()
        self.model.train()  # Restore train mode after inference
        return action

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None

        batch = self.memory.sample(batch_size)
        states, actions, next_states, rewards, dones, opponent_obs_list, next_opponent_obs_list = zip(*batch)  # Unpack done and next_opponent_obs

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)                                    # done tensor
        opponent_obs_batch = torch.FloatTensor(opponent_obs_list)
        next_opponent_obs_batch = torch.FloatTensor(next_opponent_obs_list) # next opponent obs tensor

        self.model.train()
        current_q_values = self.model(states, opponent_obs_batch)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_q_online = self.model(next_states, next_opponent_obs_batch)        # Use next_opponent_obs
                next_actions = next_q_online.argmax(1)

                next_q_target = self.target_model(next_states, next_opponent_obs_batch) # Use next_opponent_obs
                max_next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_model(next_states, next_opponent_obs_batch) # Use next_opponent_obs
                max_next_q_values = next_q_values.max(1)[0]

            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)   # Terminal state masking

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

    def remember(self, state, action, next_state, reward, done, opponent_obs, next_opponent_obs):  # Added done, next_opponent_obs
        self.memory.push(state, action, next_state, reward, done, opponent_obs, next_opponent_obs)

    def inspect(self, state, opponent_obs):
        self.model.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            o = torch.FloatTensor(opponent_obs).unsqueeze(0)
            q_final, q_experts, w = self.model(s, o, return_expert_values=True)
            q_final = q_final.squeeze(0).cpu().numpy()     # (A,)
            q_experts = q_experts.squeeze(0).cpu().numpy() # (A, K)
            w = w.squeeze(0).cpu().numpy()                 # (K,)
        self.model.train()  
        return q_final, q_experts, w
