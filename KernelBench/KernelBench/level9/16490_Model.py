import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Model, self).__init__()
        h_size_1 = 100
        h_size_2 = 100
        self.p_fc1 = nn.Linear(num_inputs, h_size_1)
        self.p_fc2 = nn.Linear(h_size_1, h_size_2)
        self.v_fc1 = nn.Linear(num_inputs, h_size_1 * 5)
        self.v_fc2 = nn.Linear(h_size_1 * 5, h_size_2)
        self.mu = nn.Linear(h_size_2, num_outputs)
        self.log_std = nn.Parameter(torch.zeros(1, num_outputs))
        self.v = nn.Linear(h_size_2, 1)
        for name, p in self.named_parameters():
            if 'bias' in name:
                p.data.fill_(0)
            """
            if 'mu.weight' in name:
                p.data.normal_()
                p.data /= torch.sum(p.data**2,0).expand_as(p.data)"""
        self.train()

    def forward(self, inputs):
        x = F.tanh(self.p_fc1(inputs))
        x = F.tanh(self.p_fc2(x))
        mu = self.mu(x)
        sigma_sq = torch.exp(self.log_std)
        x = F.tanh(self.v_fc1(inputs))
        x = F.tanh(self.v_fc2(x))
        v = self.v(x)
        return mu, sigma_sq, v


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_inputs': 4, 'num_outputs': 4}]
