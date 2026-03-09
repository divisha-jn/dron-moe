import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class EpisodeBuffer:
    def __init__(self):
        self.states        = []
        self.actions       = []
        self.rewards       = []
        self.values        = []
        self.log_probs     = []
        self.dones         = []
        self.opponent_obs  = []
        self.last_next_state    = None
        self.last_next_opp_obs  = None

    def push(self, state, action, reward, value, log_prob, done, opponent_obs):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.opponent_obs.append(opponent_obs)

    def get(self):
        return (
            np.array(self.states,       dtype=np.float32),
            np.array(self.actions,      dtype=np.int64),
            np.array(self.rewards,      dtype=np.float32),
            np.array(self.values,       dtype=np.float32),
            np.array(self.log_probs,    dtype=np.float32),
            np.array(self.dones,        dtype=np.float32),
            np.array(self.opponent_obs, dtype=np.float32),
        )

    def clear(self):
        self.states       = []
        self.actions      = []
        self.rewards      = []
        self.values       = []
        self.log_probs    = []
        self.dones        = []
        self.opponent_obs = []
        self.last_next_state   = None
        self.last_next_opp_obs = None

    def __len__(self):
        return len(self.states)


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, opponent_obs_size, num_actions, hidden_size=128):
        super(ActorCriticNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size + opponent_obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state, opponent_obs):
        x        = torch.cat([state, opponent_obs], dim=-1)
        features = self.shared(x)
        return self.actor(features), self.critic(features)

    def get_action(self, state, opponent_obs, training=True):
        action_logits, value = self.forward(state, opponent_obs)
        dist     = Categorical(logits=action_logits)
        action   = dist.sample() if training else action_logits.argmax()
        log_prob = dist.log_prob(action)
        return action.item(), value.item(), log_prob.item()


class BaselinePolicyAgent:
    def __init__(self, state_size, opponent_obs_size, num_actions,
                 lr=0.00005, gamma=0.99,
                 eps_clip=0.2, epochs=4, value_coef=0.5, entropy_coef=0.02):

        self.num_actions   = num_actions
        self.gamma         = gamma
        self.eps_clip      = eps_clip
        self.epochs        = epochs
        self.value_coef    = value_coef
        self.entropy_coef  = entropy_coef

        self.model     = ActorCriticNetwork(state_size, opponent_obs_size, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        self.loss_fn   = nn.MSELoss()  

        self.buffer         = EpisodeBuffer()
        self.training_steps = 0

        self.reward_mean  = 0.0
        self.reward_std   = 1.0
        self.reward_count = 0

    def select_action(self, state, opponent_obs, training=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        opp_tensor   = torch.FloatTensor(opponent_obs).unsqueeze(0)

        with torch.no_grad():
            action, value, log_prob = self.model.get_action(
                state_tensor, opp_tensor, training
            )

        if training:
            self.last_state        = state
            self.last_action       = action
            self.last_value        = value
            self.last_log_prob     = log_prob
            self.last_opponent_obs = opponent_obs

        return action

    def remember(self, reward, done, next_state, next_opponent_obs):
 
        self.buffer.push(
            self.last_state,
            self.last_action,
            reward,
            self.last_value,
            self.last_log_prob,
            done,
            self.last_opponent_obs
        )
        self.buffer.last_next_state   = next_state
        self.buffer.last_next_opp_obs = next_opponent_obs

    def compute_gae(self, rewards, values, dones, next_value):
        advantages  = []
        gae         = 0.0
        lambda_gae  = 0.95
        values      = np.append(values, next_value)

        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae   = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae   = delta + self.gamma * lambda_gae * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages, dtype=np.float32)
        returns    = advantages + values[:-1]
        return advantages, returns

    def train_step(self, batch_size=64):
        if len(self.buffer) == 0:
            return None

        states, actions, rewards, values, old_log_probs, dones, opponent_obs = self.buffer.get()

        if len(rewards) > 1:
            reward_mean        = rewards.mean()
            reward_std         = rewards.std() + 1e-8
            self.reward_count += len(rewards)
            alpha              = min(0.1, len(rewards) / max(self.reward_count, 1))
            self.reward_mean   = (1 - alpha) * self.reward_mean + alpha * reward_mean
            self.reward_std    = (1 - alpha) * self.reward_std  + alpha * reward_std
            rewards            = (rewards - self.reward_mean) / (self.reward_std + 1e-8)

        if dones[-1]:
            next_value = 0.0
        else:
            with torch.no_grad():
                ns = torch.FloatTensor(self.buffer.last_next_state).unsqueeze(0)
                no = torch.FloatTensor(self.buffer.last_next_opp_obs).unsqueeze(0)
                _, nv = self.model(ns, no)
                next_value = nv.item()

        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t        = torch.FloatTensor(states)
        actions_t       = torch.LongTensor(actions)
        old_log_probs_t = torch.FloatTensor(old_log_probs)
        advantages_t    = torch.FloatTensor(advantages)
        returns_t       = torch.FloatTensor(returns)
        opp_obs_t       = torch.FloatTensor(opponent_obs)

        total_loss = total_actor = total_critic = total_entropy = 0.0

        for _ in range(self.epochs):
            action_logits, values_pred = self.model(states_t, opp_obs_t)
            dist      = Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions_t)
            entropy   = dist.entropy().mean()

            ratio  = torch.exp(log_probs - old_log_probs_t)
            surr1  = ratio * advantages_t
            surr2  = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_t
            actor_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.loss_fn(values_pred.squeeze(), returns_t)  
            loss       = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            if torch.isnan(loss):
                print("NaN detected — skipping update")
                self.buffer.clear()
                return None

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_loss    += loss.item()
            total_actor   += actor_loss.item()
            total_critic  += value_loss.item()
            total_entropy += entropy.item()

        self.scheduler.step()
        self.buffer.clear()
        self.training_steps += 1

        return {
            "loss":        total_loss    / self.epochs,
            "actor_loss":  total_actor   / self.epochs,
            "critic_loss": total_critic  / self.epochs,
            "entropy":     total_entropy / self.epochs,
        }

    def inspect(self, state, opponent_obs):
        self.model.eval()
        with torch.no_grad():
            s      = torch.FloatTensor(state).unsqueeze(0)
            o      = torch.FloatTensor(opponent_obs).unsqueeze(0)
            logits, value = self.model(s, o)
            probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        self.model.train()  # Restore train mode
        return probs, float(value.item())
