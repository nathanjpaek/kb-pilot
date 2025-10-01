import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta
from abc import abstractmethod


def init_lecun_normal(tensor, scale=1.0):
    """Initializes the tensor with LeCunNormal."""
    fan_in = torch.nn.init._calculate_correct_fan(tensor, 'fan_in')
    std = scale * np.sqrt(1.0 / fan_in)
    with torch.no_grad():
        return tensor.normal_(0, std)


@torch.no_grad()
def init_chainer_default(layer):
    """Initializes the layer with the chainer default.
    weights with LeCunNormal(scale=1.0) and zeros as biases
    """
    assert isinstance(layer, nn.Module)
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        init_lecun_normal(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return layer


class MLP(nn.Module):
    """Multi-Layer Perceptron"""

    def __init__(self, in_size, out_size, hidden_sizes, nonlinearity=F.relu,
        last_wscale=1):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        super().__init__()
        if hidden_sizes:
            self.hidden_layers = nn.ModuleList()
            self.hidden_layers.append(nn.Linear(in_size, hidden_sizes[0]))
            for hin, hout in zip(hidden_sizes, hidden_sizes[1:]):
                self.hidden_layers.append(nn.Linear(hin, hout))
            self.hidden_layers.apply(init_chainer_default)
            self.output = nn.Linear(hidden_sizes[-1], out_size)
        else:
            self.output = nn.Linear(in_size, out_size)
        init_lecun_normal(self.output.weight, scale=last_wscale)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        h = x
        if self.hidden_sizes:
            for l in self.hidden_layers:
                h = self.nonlinearity(l(h))
        return self.output(h)


class StateActionQFunction(object, metaclass=ABCMeta):
    """Abstract Q-function with state and action input."""

    @abstractmethod
    def __call__(self, x, a):
        """Evaluates Q-function

        Args:
            x (ndarray): state input
            a (ndarray): action input

        Returns:
            Q-value for state x and action a
        """
        raise NotImplementedError()


class FCLateActionSAQFunction(nn.Module, StateActionQFunction):
    """Fully-connected (s,a)-input Q-function with late action input.

    Actions are not included until the second hidden layer and not normalized.
    This architecture is used in the DDPG paper:
    http://arxiv.org/abs/1509.02971

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers. It must be greater than
            or equal to 1.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
        n_hidden_layers, nonlinearity=F.relu, last_wscale=1.0):
        assert n_hidden_layers >= 1
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.nonlinearity = nonlinearity
        super().__init__()
        self.obs_mlp = MLP(in_size=n_dim_obs, out_size=n_hidden_channels,
            hidden_sizes=[])
        self.mlp = MLP(in_size=n_hidden_channels + n_dim_action, out_size=1,
            hidden_sizes=[self.n_hidden_channels] * (self.n_hidden_layers -
            1), nonlinearity=nonlinearity, last_wscale=last_wscale)
        self.output = self.mlp.output

    def forward(self, state, action):
        h = self.nonlinearity(self.obs_mlp(state))
        h = torch.cat((h, action), dim=1)
        return self.mlp(h)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_dim_obs': 4, 'n_dim_action': 4, 'n_hidden_channels': 4,
        'n_hidden_layers': 1}]
