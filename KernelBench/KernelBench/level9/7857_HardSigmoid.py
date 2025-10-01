import torch
from torch import nn


class HardSigmoid(nn.Module):

    def __init__(self, slope=0.1666667, offset=0.5):
        super().__init__()
        self.slope = slope
        self.offset = offset

    def forward(self, input):
        return torch.where(input >= 3, torch.tensor(1.0, dtype=torch.float)
            .type_as(input), torch.where(input <= -3, torch.tensor(0.0,
            dtype=torch.float).type_as(input), torch.add(torch.mul(input,
            self.slope), self.offset)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
