import torch
from torch import nn


class WeightedBCE(nn.Module):

    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, inputs, targets):
        inputs = inputs.view(-1).float()
        targets = targets.view(-1).float()
        if self.weights is not None:
            assert len(self.weights) == 2
            loss = weights[1] * (targets * torch.log(inputs)) + weights[0] * ((
                1 - targets) * torch.log(1 - inputs))
        else:
            loss = targets * torch.log(inputs) + (1 - targets) * torch.log(
                1 - inputs)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
