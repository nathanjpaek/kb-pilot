import torch
from typing import List
from typing import Tuple
import torch.nn as nn


class LinearNetwork(nn.Module):

    def __init__(self, input_size: 'int', output_size: 'int', hidden_layers:
        'List[int]', activation: 'nn.Module'=nn.LeakyReLU):
        super(LinearNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.n_layers = len(hidden_layers)
        self.layers = nn.Sequential()
        if self.n_layers == 0:
            self.layers.add_module('single_layer', nn.Linear(input_size,
                output_size))
        else:
            for i in range(self.n_layers + 1):
                if i == 0:
                    self.layers.add_module('input_layer', nn.Linear(
                        input_size, hidden_layers[0]))
                    self.layers.add_module('input_layer_activation', self.
                        activation())
                elif i < self.n_layers:
                    self.layers.add_module(f'hidden_layer_{i}', nn.Linear(
                        hidden_layers[i - 1], hidden_layers[i]))
                    self.layers.add_module(f'input_layer_{i}_activation',
                        self.activation())
                else:
                    self.layers.add_module('output_layer', nn.Linear(
                        hidden_layers[i - 1], output_size))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        assert len(x.shape) == 2
        assert x.shape[1] == self.input_size
        return self.layers(x)


class TanhGaussianPolicy(nn.Module):
    LOG_STD_MIN: 'float' = -20.0
    LOG_STD_MAX: 'float' = 2.0
    EPS: 'float' = 1e-06

    def __init__(self, n_agents: 'int', obs_size: 'int', act_size: 'int',
        hidden_layers: 'List[int]', activation: 'nn.Module'=nn.LeakyReLU):
        super(TanhGaussianPolicy, self).__init__()
        self.n_agents = n_agents
        self.obs_size = obs_size
        self.act_size = act_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.input_size = obs_size
        self.output_size = 2 * act_size
        self.policy = LinearNetwork(self.input_size, self.output_size,
            hidden_layers, activation)

    def forward(self, obs: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        assert len(obs.shape) == 2
        assert obs.shape[1] == self.obs_size
        obs.shape[0]
        mean, log_std = torch.chunk(self.policy(obs), 2, dim=-1)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.
            LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: 'torch.Tensor') ->Tuple[torch.Tensor, torch.
        Tensor, torch.Tensor]:
        assert len(obs.shape) == 3
        assert obs.shape[1] == self.n_agents
        assert obs.shape[2] == self.obs_size
        N = obs.shape[0]
        means = torch.empty((N, self.n_agents, self.act_size))
        stds = torch.empty((N, self.n_agents, self.act_size))
        for i in range(self.n_agents):
            mean, log_std = self.forward(obs[:, i])
            std = log_std.exp()
            means[:, i] = mean
            stds[:, i] = std
        dist = torch.distributions.normal.Normal(means, stds)
        act_sampled = dist.rsample()
        act_sampled_tanh = torch.tanh(act_sampled)
        log_probs = dist.log_prob(act_sampled) - torch.log(1 -
            act_sampled_tanh.square() + self.EPS)
        entropies = -log_probs.sum(dim=-1, keepdim=False)
        return act_sampled_tanh, entropies, torch.tanh(means)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_agents': 4, 'obs_size': 4, 'act_size': 4,
        'hidden_layers': [4, 4]}]
