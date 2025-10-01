import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque


class act_model(nn.Module):

    def __init__(self, inp, hidden, output):
        super(act_model, self).__init__()
        self.fc1 = nn.Linear(inp, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, output, bias=True)
        self.fc12 = nn.LeakyReLU()
        self.memory = deque(maxlen=500)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        self.mse = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def action(self, state):
        if random.random() <= self.epsilon:
            return np.random.choice(out, 1)[0]
        else:
            q_values = self.forward(state)
            return np.argmax(q_values.detach().numpy())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                q_values = self.forward(next_state)
                target = reward + self.gamma * np.amax(q_values.detach().
                    numpy())
            target_f = self.forward(state)
            target_f[action] = target
            target_g = self.forward(state)
            self.zero_grad()
            self.optimizer.zero_grad()
            loss = self.mse(target_g, target_f)
            loss.backward(retain_graph=True)
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, PATH):
        torch.save(self, PATH)

    def save(self, PATH):
        model = torch.load(PATH)
        return model

    def forward(self, x):
        out = self.fc12(self.fc1(x))
        out = self.fc12(self.fc2(out))
        out = self.fc3(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inp': 4, 'hidden': 4, 'output': 4}]
