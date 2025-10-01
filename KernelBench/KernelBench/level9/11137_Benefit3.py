import torch
import torch.nn as nn


class Benefit3(nn.Module):

    def __init__(self):
        super(Benefit3, self).__init__()
        self.delta = torch.nn.Parameter(torch.FloatTensor([0.03]),
            requires_grad=True)

    def forward(self, I, A, B):
        self.Y = I * self.delta + A * self.delta ** 2 + B * self.delta ** 3
        return self.Y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
