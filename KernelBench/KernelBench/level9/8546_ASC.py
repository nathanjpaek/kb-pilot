import torch
import torch.optim
import torch.nn as nn
import torch.nn.init


class ASC(nn.Module):

    def __init__(self, a=3.5):
        super().__init__()
        self.a = a

    def forward(self, input):
        return torch.div(torch.exp(self.a * input), torch.sum(torch.exp(
            self.a * input), dim=1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
