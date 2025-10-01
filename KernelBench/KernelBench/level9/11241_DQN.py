import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DQN(nn.Module):

    def __init__(self, state_dim, out_dim, capacity, bsz, epsilon):
        super().__init__()
        self.steps_done = 0
        self.position = 0
        self.pool = []
        self.capacity = capacity
        self.bsz = bsz
        self.epsilon = epsilon
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, out_dim)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def action(self, state):
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        if random.random() > self.epsilon:
            return self(Variable(state, volatile=True)).data.max(1)[1].view(
                1, 1)
        else:
            return longTensor([[random.randrange(2)]])

    def push(self, *args):
        if len(self) < self.capacity:
            self.pool.append(None)
        self.pool[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return random.sample(self.pool, self.bsz)

    def __len__(self):
        return len(self.pool)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'out_dim': 4, 'capacity': 4, 'bsz': 4,
        'epsilon': 4}]
