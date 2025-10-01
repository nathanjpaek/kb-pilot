import torch
import torch.nn as nn


class Myloss(nn.Module):

    def __init__(self, epsilon=1e-08):
        super(Myloss, self).__init__()
        self.epsilon = epsilon
        return

    def forward(self, input_, label, weight):
        entropy = -label * torch.log(input_ + self.epsilon) - (1 - label
            ) * torch.log(1 - input_ + self.epsilon)
        return torch.sum(entropy * weight) / 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
