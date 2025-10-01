import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):

    def __init__(self, n_inputs, n_outputs, n_hidden=64, lr=0.001, softmax=
        False, device='cpu'):
        super(Model, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.softmax = softmax
        self.fc1 = nn.Linear(self.n_inputs, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_outputs)
        self.optimizer = optim.Adam(self.parameters(), lr)
        self.device = device
        self

    def forward(self, x):
        h_relu = F.relu(self.fc1(x))
        y = self.fc2(h_relu)
        if self.softmax:
            y = F.softmax(self.fc2(h_relu), dim=-1).clamp(min=1e-09, max=1 -
                1e-09)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_inputs': 4, 'n_outputs': 4}]
