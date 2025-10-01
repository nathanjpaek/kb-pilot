import torch
import numpy as np
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, num_distros):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, num_distros)

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        return out

    def get_mixture_coef(self, x):
        y = self.forward(x)
        out_pi, out_sigma, out_mu = np.split(y, 3)
        max_pi = np.amax(out_pi)
        out_pi = np.subtract(out_pi, max_pi)
        out_pi = np.exp(out_pi)
        normalize_pi = np.reciprocal(np.sum(out_pi, axis=1))
        out_pi = np.multiply(normalize_pi, out_pi)
        out_sigma = np.exp(out_sigma)
        return out_pi, out_sigma, out_mu


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'num_distros': 4}]
