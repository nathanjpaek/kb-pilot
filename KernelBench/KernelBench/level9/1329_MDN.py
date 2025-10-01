import torch
import torch.nn as nn
import torch.nn.functional as F


class MDN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(MDN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.z_h = nn.Linear(input_size, hidden_size)
        self.z_pi = nn.Linear(hidden_size, output_size)
        self.z_mu = nn.Linear(hidden_size, output_size)
        self.z_sig = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        z_h = F.tanh(self.z_h(inp))
        pi = F.softmax(self.z_pi(z_h))
        mu = self.z_mu(z_h)
        sig = torch.exp(self.z_sig(z_h))
        return pi, sig, mu


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'output_size': 4,
        'batch_size': 4}]
