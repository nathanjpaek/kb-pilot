import torch
import numpy as np
from abc import ABC
from abc import abstractmethod
import torch.nn.functional as F
from torch.functional import F
from torch import nn
from typing import *
from torch.nn import functional as F


def to_array_as(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, np.ndarray):
        return x.detach().cpu().numpy().astype(y.dtype)
    elif isinstance(x, np.ndarray) and isinstance(y, torch.Tensor):
        return torch.tensor(x)
    else:
        return x


class BasePolicy(ABC):

    @abstractmethod
    def policy_infer(self, obs):
        pass

    def get_action(self, obs):
        obs_tensor = torch.tensor(obs, device=next(self.parameters()).
            device, dtype=torch.float32)
        act = to_array_as(self.policy_infer(obs_tensor), obs)
        return act


class VAE(nn.Module, BasePolicy):

    def __init__(self, state_dim, action_dim, latent_dim, max_action,
        hidden_size=750):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)
        self.d1 = nn.Linear(state_dim + latent_dim, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)
        self.d3 = nn.Linear(hidden_size, action_dim)
        self.max_action = max_action
        self.latent_dim = latent_dim
        self._actor = None

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))
        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        u = self.decode(state, z)
        return u, mean, std

    def decode(self, state, z=None, clip=None, raw=False):
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim))
            if clip is not None:
                z = z.clamp(-clip, clip)
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        a = self.d3(a)
        if raw:
            return a
        return self.max_action * torch.tanh(a)

    def policy_infer(self, obs):
        return self.decode(obs, z=self._actor(obs)[0])


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4, 'latent_dim': 4,
        'max_action': 4}]
