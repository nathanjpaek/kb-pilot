import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class BasicNet:

    def __init__(self, optimizer_fn, gpu, LSTM=False):
        self.gpu = gpu and torch.cuda.is_available()
        self.LSTM = LSTM
        if self.gpu:
            self
            self.FloatTensor = torch.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

    def to_torch_variable(self, x, dtype='float32'):
        if isinstance(x, Variable):
            return x
        if not isinstance(x, torch.FloatTensor):
            x = torch.from_numpy(np.asarray(x, dtype=dtype))
        if self.gpu:
            x = x
        return Variable(x)

    def reset(self, terminal):
        if not self.LSTM:
            return
        if terminal:
            self.h.data.zero_()
            self.c.data.zero_()
        self.h = Variable(self.h.data)
        self.c = Variable(self.c.data)


class GaussianActorNet(nn.Module, BasicNet):

    def __init__(self, state_dim, action_dim, action_scale=1.0, action_gate
        =None, gpu=False, unit_std=True, hidden_size=64):
        super(GaussianActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.action_mean = nn.Linear(hidden_size, action_dim)
        if unit_std:
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        else:
            self.action_std = nn.Linear(hidden_size, action_dim)
        self.unit_std = unit_std
        self.action_scale = action_scale
        self.action_gate = action_gate
        BasicNet.__init__(self, None, gpu, False)

    def forward(self, x):
        x = self.to_torch_variable(x)
        phi = F.tanh(self.fc1(x))
        phi = F.tanh(self.fc2(phi))
        mean = self.action_mean(phi)
        if self.action_gate is not None:
            mean = self.action_scale * self.action_gate(mean)
        if self.unit_std:
            log_std = self.action_log_std.expand_as(mean)
            std = log_std.exp()
        else:
            std = F.softplus(self.action_std(phi)) + 1e-05
            log_std = std.log()
        return mean, std, log_std

    def predict(self, x):
        return self.forward(x)

    def log_density(self, x, mean, log_std, std):
        var = std.pow(2)
        log_density = -(x - mean).pow(2) / (2 * var + 1e-05) - 0.5 * torch.log(
            2 * Variable(torch.FloatTensor([np.pi])).expand_as(x)) - log_std
        return log_density.sum(1)

    def entropy(self, std):
        return 0.5 * (1 + (2 * std.pow(2) * np.pi + 1e-05).log()).sum(1).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4}]
