import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):

    def __init__(self, n_features=4, n_actions=2, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.device = device

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1 if x.dim() == 2 else 0)
        return x

    def get_action(self, x, mode='train'):
        action_prob = self.forward(x)
        m = Categorical(probs=action_prob)
        if mode == 'train':
            action = m.sample()
        else:
            action = torch.argmax(m.probs)
        return action.cpu().numpy()

    def get_action_with_action_prob(self, x, mode='train'):
        action_prob = self.forward(x)
        m = Categorical(probs=action_prob)
        if mode == 'train':
            action = m.sample()
            action_prob_selected = action_prob[action]
        else:
            action = torch.argmax(m.probs, dim=1 if action_prob.dim() == 2 else
                0)
            action_prob_selected = None
        return action.cpu().numpy(), action_prob_selected


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
