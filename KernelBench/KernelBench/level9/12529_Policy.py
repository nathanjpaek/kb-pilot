import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch.autograd import Variable
from torch.distributions import Categorical


class Policy(nn.Module):

    def __init__(self, in_sz, hidden_sz, out_sz):
        super(Policy, self).__init__()
        self.fc1 = Linear(in_sz, hidden_sz)
        self.fc2 = Linear(hidden_sz, out_sz)
        self.log_probs = list()
        self.rewards = list()

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = F.relu(x)
        logits = self.fc2(x)
        return F.softmax(logits, dim=0)

    def act(self, inputs):
        if torch.cuda.is_available():
            inputs = Variable(Tensor(inputs))
        else:
            inputs = Variable(Tensor(inputs))
        probs = self(inputs)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.data[0]

    def learn(self, optimizer):
        self.weight_reward()
        losses = []
        for log_prob, reward in zip(self.log_probs, self.rewards):
            losses.append(-log_prob * reward)
        optimizer.zero_grad()
        losses = torch.cat(losses).sum()
        losses.backward()
        optimizer.step()
        self.rewards = list()
        self.log_probs = list()

    def weight_reward(self):
        R = 0
        rewards = []
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            rewards.insert(0, R)
        rewards = Tensor(rewards)
        self.rewards = (rewards - rewards.mean()) / (rewards.std() + np.
            finfo(np.float32).eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_sz': 4, 'hidden_sz': 4, 'out_sz': 4}]
