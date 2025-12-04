

import sys
import gym
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, num_atoms):
        super(Critic, self).__init__()

        self.h_linear_1 = nn.Linear(in_features=input_size, out_features=256)
        self.h_linear_2 = nn.Linear(in_features=256, out_features=128)
        self.h_linear_3 = nn.Linear(in_features=128, out_features=num_atoms)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenates the seq tensors in the given dimension
        x = torch.relu(self.h_linear_1(x))
        x = torch.relu(self.h_linear_2(x))
        x = self.h_linear_3(x)  # No activation function here
        x = F.softmax(x, dim=1)  # softmax because critic should output probabilities
        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()

        self.h_linear_1 = nn.Linear(input_size, 256)
        self.h_linear_2 = nn.Linear(256, 128)
        self.h_linear_3 = nn.Linear(128, output_size)

    def forward(self, state):
        x = torch.relu(self.h_linear_1(state))
        x = torch.relu(self.h_linear_2(x))
        x = torch.tanh(self.h_linear_3(x))
        return x


class NoiseGenerator:
    def __init__(self, action_dims, action_bound_high, noise_scale=0.3):

        self.action_dims = action_dims
        self.action_bounds = action_bound_high
        self.noise_scale = noise_scale

    def noise_gen(self):
        noise = np.random.normal(size=self.action_dims) * self.action_bounds * self.noise_scale
        return noise


class Memory:
    def __init__(self, replay_max_size):
        self.replay_max_size = replay_max_size
        self.replay_buffer = deque(maxlen=replay_max_size)  # batch of experiences to sample during training

    def replay_buffer_add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.replay_buffer.append(experience)

    def sample_experience(self, batch_size):
        state_batch  = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.replay_buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.replay_buffer)


class PerMemory(object):
    # stored as ( state, action, reward, next_state ) in SumTree
    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.4

    PER_b_increment_per_sampling = 0.001
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        self.tree = SumTree(capacity)

    def per_add(self, state, action, reward, next_state, done):
        experience = state, action, (reward), next_state, done
        self.store(experience)

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max priority for new priority

    def sample_experience(self, n):
        minibatch = []
        b_idx = np.empty((n,), dtype=np.int32)

        priority_segment = self.tree.total_priority / n  # priority segment

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(value)
            b_idx[i] = index
            minibatch.append([data[0], data[1], data[2], data[3], data[4]])

        return b_idx, minibatch

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity

        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


class D4PGAgent:

    def __init__(self, env, actor_learning_rate=1e-4, critic_learning_rate=1e-4, gamma=0.99,
                 max_memory_size=50000, tau=1e-3, n_steps=1):

        self.num_states  = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]

        self.act_max_bound = env.action_space.high
        self.act_min_bound = env.action_space.low

        self.n_atoms = 51
        self.v_min   = -10
        self.v_max   = 10
        self.delta = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.v_lin = torch.linspace(self.v_min, self.v_max, self.n_atoms).view(-1, 1)

        self.gamma = gamma
        self.tau = tau
        self.n_steps = n_steps
        self.t_step = 0

        hidden_size = 256
        self.actor  = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.n_atoms)

        self.actor_target  = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.n_atoms)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self.noise = NoiseGenerator(self.num_actions, self.act_max_bound)

        self.memory = Memory(max_memory_size)
        per_max_memory_size = 10000
        self.memory_per = PerMemory(per_max_memory_size)

        # logging tiện cho bên ngoài nếu cần
        self.last_actor_loss = None
        self.last_critic_loss = None
        self.did_learn_last_step = False

    def get_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor)
            action = action.detach()
            action = action.numpy()
            noise = np.random.normal(size=action.shape)
            action = np.clip(action + noise, -1, 1)
        self.actor.train()
        return action[0]

    def distr_projection(self, next_distribution, rewards, dones):
        next_distr = next_distribution.data.cpu().numpy()
        rewards    = rewards.data.cpu().numpy()
        dones_mask = dones.cpu().numpy().astype(bool)
        batch_size = len(rewards)
        proj_distr = np.zeros((batch_size, self.n_atoms), dtype=np.float32)
        gamma      = self.gamma ** self.n_steps

        for atom in range(self.n_atoms):
            tz_j = np.minimum(self.v_max, np.maximum(self.v_min, rewards + (self.v_min + atom * self.delta) * gamma))
            b_j = (tz_j - self.v_min) / self.delta
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
            ne_mask = u != l
            proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

        if dones_mask.any():
            proj_distr[dones_mask] = 0.0
            tz_j = np.minimum(self.v_max, np.maximum(self.v_min, rewards[dones_mask]))
            b_j = (tz_j - self.v_min) / self.delta
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            eq_dones = dones_mask.copy()
            eq_dones[dones_mask] = eq_mask
            if eq_dones.any():
                proj_distr[eq_dones, l[eq_mask]] = 1.0
            ne_mask = u != l
            ne_dones = dones_mask.copy()
            ne_dones[dones_mask] = ne_mask
            if ne_dones.any():
                proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
                proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]

        return torch.FloatTensor(proj_distr)

    def step_training(self, state, action, reward, next_state, done, batch_size, per_memory_status):
        # Save experience in memory
        if per_memory_status:
            self.memory_per.per_add(state, action, reward, next_state, done)
        else:
            self.memory.replay_buffer_add(state, action, reward, next_state, done)

        # Học thường xuyên hơn: mỗi bước 1 lần (nếu đủ batch)
        LEARN_EVERY_STEP = 1
        self.t_step = self.t_step + 1

        if self.t_step % LEARN_EVERY_STEP == 0:
            self.learn_step(batch_size, per_memory_status)

    def learn_step(self, batch_size, per_memory_status):
        self.did_learn_last_step = False

        if per_memory_status:
            tree_idx, minibatch = self.memory_per.sample_experience(batch_size)
            states = np.zeros((batch_size, self.num_states))
            next_states = np.zeros((batch_size, self.num_states))
            actions, rewards, dones = [], [], []
            for i in range(batch_size):
                states[i] = minibatch[i][0]
                actions.append(minibatch[i][1])
                rewards.append(minibatch[i][2])
                next_states[i] = minibatch[i][3]
                dones.append(minibatch[i][4])
        else:
            if self.memory.__len__() <= batch_size:
                return
            else:
                states, actions, rewards, next_states, dones = self.memory.sample_experience(batch_size)

        states  = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones   = np.array(dones)
        next_states = np.array(next_states)

        states  = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones   = torch.ByteTensor(dones)
        next_states = torch.FloatTensor(next_states)

        # calculate the next Z distribution Z(s',a')
        next_actions = self.actor_target.forward(next_states)
        next_Z_val   = self.critic_target.forward(next_states, next_actions.detach())

        # project target distribution
        proj_distr_v = self.distr_projection(next_Z_val, rewards, dones)
        Y = proj_distr_v

        # current distribution Z(s,a)
        Z_val = self.critic.forward(states, actions)

        # critic loss
        BCE_loss = torch.nn.BCELoss(reduction='none')
        td_error = BCE_loss(Z_val, Y)
        td_error = td_error.mean(axis=1)
        critic_loss = td_error.mean()

        # actor loss
        z_atoms = np.linspace(self.v_min, self.v_max, self.n_atoms)
        z_atoms = torch.from_numpy(z_atoms).float()
        actor_probs = self.critic.forward(states, self.actor.forward(states))
        actor_loss = actor_probs * z_atoms
        actor_loss = torch.sum(actor_loss, dim=1)
        actor_loss = -actor_loss.mean()

        # Update priorities for PER
        if per_memory_status:
            td_error_np = td_error.detach().numpy().flatten()
            absolute_errors = np.abs(td_error_np)
            self.memory_per.batch_update(tree_idx, absolute_errors)

        # Actor update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # soft-update targets
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        self.last_actor_loss = float(actor_loss.detach().cpu().numpy())
        self.last_critic_loss = float(critic_loss.detach().cpu().numpy())
        self.did_learn_last_step = True


def main():
    EPISODES = 500
    batch_size = 64
    rollout_steps = 1
    gamma = 0.99

    env = gym.make('Pendulum-v1')
    agent = D4PGAgent(env, gamma=gamma, n_steps=rollout_steps)

    per_memory_status = False
    rewards = []
    avg_rewards = []

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            env.render()
            n_step_reward = 0
            for n in range(rollout_steps):
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                n_step_reward += reward * gamma ** n
                if n == (rollout_steps - 1):
                    agent.step_training(state, action, n_step_reward, next_state, done, batch_size, per_memory_status)
                state = next_state
                episode_reward += reward
            if done:
                print(episode_reward, episode)
                break

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))

    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


if __name__ == '__main__':
    main()

