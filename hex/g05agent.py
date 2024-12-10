import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.core.multiarray import ndarray

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(float):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, in_size, out_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(in_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, out_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x


class G05Agent:
    def __init__(self, env, training_mode=False):
        self.env = env
        self.training_mode = training_mode

        # setting up nn
        in_size, out_size = self.nn_size()
        self.target_net = DQN(in_size, out_size).to(device)
        self.policy_net = DQN(in_size, out_size).to(device)
        # self.target_net.load_state_dict(self.policy_net.state_dict())

        # learning parameters
        self.gamma = 0.99

        self.eps_decay = 0.995
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps = self.eps_start

        self.training_step = 0
        self.replay_buffer = ReplayMemory(100)
        self.batch_size = 10
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)

    def update(self):
        if self.replay_buffer.__len__() < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.stack(batch.state).to(device)
        actions = torch.stack(batch.action).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        rewards = torch.stack(batch.reward).to(device)

        q_values = self.policy_net.forward(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net.forward(next_states).max(1)[0]

        target_q_values = rewards.squeeze(1) + (self.gamma * next_q_values)

        criterion = nn.MSELoss()
        loss = criterion(q_values.squeeze(1), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        if self.training_step % 20 == 0:
            self.policy_net = self.target_net
            self.save_model()

    def nn_input_and_mask(self):
        observation, reward, termination, truncation, info = self.env.last()
        return self.process_environment(observation, info)

    def decay_epsilon(self, nn_input, mask) -> int:
        action = 0
        if random.random() < self.eps:
            valid_actions = []
            for i in range(len(mask)):
                if mask[i] != 0:
                    valid_actions.append(i)

            action = random.choice(valid_actions)
        else:
            action = self.best_action(nn_input, mask)

        self.eps = max(self.eps * self.eps_decay, self.eps_end)
        return action

    def train(self):
        self.training_step += 1
        agent_name = self.env.agent_selection
        nn_input, mask = self.nn_input_and_mask()
        action = self.decay_epsilon(nn_input, mask)

        self.env.step(action)

        observation, cumulative_reward, termination, truncation, info = self.env.last()
        immediate_reward = self.env.rewards[agent_name]

        color = "red" if agent_name == "player_1" else "blue"
        print(
            f"{color}, action: {action}, cumulative_reward: {cumulative_reward}, terminated: {termination or truncation}")

        if termination or truncation:
            eps = self.eps_start

        new_nn_input, new_mask = self.process_environment(observation, info)
        self.replay_buffer.push(nn_input, torch.tensor([action]), new_nn_input, torch.tensor([cumulative_reward]))
        self.update()

    def nn_size(self) -> (int, int):
        broad_size = self.env.board_size

        # that's two seperated broad with pie rule
        in_size = 2 * broad_size ** 2 + 2
        out_size = broad_size ** 2 + 1
        return in_size, out_size

    def process_environment(self, observation, info) -> (ndarray, ndarray):
        pie_rule_used = observation["pie_rule_used"]
        broads = observation["observation"]
        mask = info['action_mask']
        direction = info["direction"]

        broads = broads.flatten()

        # below take
        # two parts (direction, pi_rule, direction) -> compressed into one and mask
        # either do some input processing or directly feed in raw
        board1 = np.zeros(self.env.board_size ** 2)
        board2 = np.zeros(self.env.board_size ** 2)

        # divide into two broads
        for i in range(self.env.board_size ** 2):
            stone = broads[i]
            if stone == 0:
                continue
            elif stone == 1:
                board1[i] = 1
            elif stone == 2:
                board2[i] = 1

        nn_input = np.concatenate((np.array([pie_rule_used]), np.array([direction]), board1, board2))
        # np.array(pie_rule_used)

        # if direction:
        #     nn_input = np.append(nn_input, board1)
        #     nn_input = np.append(nn_input, board2)
        # else:
        #     nn_input = np.append(nn_input, board2)
        #     nn_input = np.append(nn_input, board1)

        nn_input = torch.from_numpy(nn_input).float().to(device)
        return nn_input, mask

    def best_action(self, nn_input, mask) -> int:
        nn_output = self.policy_net.forward(nn_input)
        nn_output = nn_output.cpu().detach().numpy()
        # this assumes two array have the same dimension, if at index i mask[i] == 0: out[i] = -1e9
        nn_output[mask == 0] = -1e9

        return np.argmax(nn_output)

    def select_action(self, observation, reward, termination, truncation, info):
        if self.training_mode:
            self.train()

        else:
            self.load_model()
            nn_input, mask = self.process_environment(observation, info)
            return self.best_action(nn_input, mask)

    def load_model(self, PATH="policy_net.pth"):
        self.policy_net.load_state_dict(torch.load(PATH))

    def save_model(self, PATH="policy_net.pth"):
        torch.save(self.policy_net.state_dict(), PATH)
