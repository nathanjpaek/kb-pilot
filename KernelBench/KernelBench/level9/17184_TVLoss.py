import torch
import torch as th
import torch.utils.data
import torch
import torch.autograd


class TVLoss(th.nn.Module):

    def __init__(self, strength=1.0):
        super(TVLoss, self).__init__()
        self.strength = strength

    def forward(self, input):
        self.x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        self.y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        self.loss = (th.sum(th.abs(self.x_diff)) + th.sum(th.abs(self.y_diff))
            ) * self.strength
        return input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
