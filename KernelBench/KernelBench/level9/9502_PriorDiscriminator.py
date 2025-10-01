import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class PriorDiscriminator(nn.Module):
    """The prior discriminator class.

    This discriminate between a vector drawn from random uniform,
    and the vector y obtained as output of the encoder.
    It enforces y to be close to a uniform distribution.
    """

    def __init__(self, y_size):
        super().__init__()
        self.l0 = nn.Linear(y_size, 512)
        self.l1 = nn.Linear(512, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'y_size': 4}]
