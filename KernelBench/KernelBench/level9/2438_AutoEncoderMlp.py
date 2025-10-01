import abc
import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.utils.data


class PyTorchModule(nn.Module, metaclass=abc.ABCMeta):
    """
    Keeping wrapper around to be a bit more future-proof.
    """
    pass


class AutoEncoderMlp(PyTorchModule):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 256)
        self.e2 = nn.Linear(256, 256)
        self.r1 = nn.Linear(256, 1, bias=False)
        self.a1 = nn.Linear(256, 256)
        self.a2 = nn.Linear(256, action_dim)
        self.d1 = nn.Linear(256, 256)
        self.d2 = nn.Linear(256, state_dim)
        self

    def forward(self, obs, action):
        x = F.relu(self.e1(torch.cat([obs, action], axis=1)))
        x = F.relu(self.e2(x))
        reward_prediction = self.r1(x)
        action_rec = F.relu(self.a1(x))
        action_rec = self.a2(action_rec)
        next_state_prediction = F.relu(self.d1(x))
        next_state_prediction = self.d2(next_state_prediction)
        return next_state_prediction, action_rec, reward_prediction

    def latent(self, obs, action):
        x = F.relu(self.e1(torch.cat([obs, action], axis=1)))
        x = F.relu(self.e2(x))
        return x


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4}]
