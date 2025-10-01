import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


class StoppingNetwork(nn.Module):
    """The stopping network.

    Uses the internal state `h_t` of the core network
    to determine whether the network integrated enough
    information to make a confident classification.

    Practically, take the internal state `h_t` as input
    and outputs a binary action `a_t` which states
    whether the network should stop or continue with
    the glimpses.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        h_t: the hidden state vector of the core network
            for the current time step `t`.

    Returns:
        a_t: a 2D vector of shape (B, 1).
         The stopping action for the current time step `t`.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, input_size)
        self.fc_at = nn.Linear(input_size, 1)

    def forward(self, h_t):
        feat = F.relu(self.fc(h_t.detach()))
        a_t_pi = F.hardtanh(input=self.fc_at(feat), min_val=-10, max_val=10)
        a_t_pi = torch.sigmoid(a_t_pi)
        a_t = Bernoulli(probs=a_t_pi).sample()
        a_t = a_t.detach()
        log_pi = Bernoulli(probs=a_t_pi).log_prob(a_t)
        return log_pi, a_t


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
