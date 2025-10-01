import torch
import torch.nn as nn


def tensor_gradient(input):
    input0 = input[..., :-1, :-1]
    didy = input[..., 1:, :-1] - input0
    didx = input[..., :-1, 1:] - input0
    return torch.cat((didy, didx), -3)


class GradientLoss(nn.Module):

    def forward(self, input, target):
        return torch.abs(tensor_gradient(input) - tensor_gradient(target)
            ).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
