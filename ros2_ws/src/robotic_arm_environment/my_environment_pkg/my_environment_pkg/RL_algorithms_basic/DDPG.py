
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------- Models ---------------- #

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(Critic, self).__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        nn.init.kaiming_uniform_(self.h1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.h2.weight, nonlinearity='relu')
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = self.out(x)
        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        nn.init.kaiming_uniform_(self.h1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.h2.weight, nonlinearity='relu')
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.h1(state))
        x = F.relu(self.h2(x))
        x = torch.tanh(self.out(x))
        return x


# ------------- OU Noise ------------- #

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.05, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = int(np.prod(action_space.shape))
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        a = action + ou_state
        return np.clip(a, self.low, self.high)


# ------------- Replay ------------- #

class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, np.array([reward], dtype=np.float32), next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for s, a, r, ns, d in batch:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ------------- Agent ------------- #

class DDPGagent:
    def __init__(self, env, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3,
                 gamma=0.99, tau=1e-2, max_memory_size=50000, device=None):

        self.env = env
        self.num_states = int(np.prod(env.observation_space.shape))
        self.num_actions = int(np.prod(env.action_space.shape))

        self.gamma = gamma
        self.tau = tau
        self.t_step = 0

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(self.num_states, hidden_size, self.num_actions).to(self.device)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, 1).to(self.device)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions).to(self.device)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, 1).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.memory = Memory(max_memory_size)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        # logging
        self.last_actor_loss = None
        self.last_critic_loss = None
        self.did_learn_last_step = False

    def get_action(self, state_np):
        state_t = torch.from_numpy(np.array(state_np, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        return action

    def step_training(self, batch_size, learn_every=5):
        self.t_step += 1
        self.did_learn_last_step = False
        if self.t_step % learn_every != 0:
            return
        if len(self.memory) <= batch_size:
            return
        self.learn_step(batch_size)
        self.did_learn_last_step = True

    def learn_step(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_Q = self.critic_target(next_states, next_actions)
            Q_target = rewards + self.gamma * next_Q

        Q_vals = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_vals, Q_target)

        actor_loss = - self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
                tp.data.copy_(p.data * self.tau + tp.data * (1.0 - self.tau))
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.copy_(p.data * self.tau + tp.data * (1.0 - self.tau))

        self.last_actor_loss = float(actor_loss.detach().cpu().item())
        self.last_critic_loss = float(critic_loss.detach().cpu().item())

    # ---------- save / load ----------
    def save(self, path_prefix: str):
        torch.save(self.actor.state_dict(),  f"{path_prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pt")
        torch.save(self.actor_target.state_dict(),  f"{path_prefix}_actor_t.pt")
        torch.save(self.critic_target.state_dict(), f"{path_prefix}_critic_t.pt")

    def load(self, path_prefix: str, map_location=None):
        map_location = map_location or self.device
        self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pt", map_location=map_location))
        self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pt", map_location=map_location))
        self.actor_target.load_state_dict(torch.load(f"{path_prefix}_actor_t.pt", map_location=map_location))
        self.critic_target.load_state_dict(torch.load(f"{path_prefix}_critic_t.pt", map_location=map_location))

