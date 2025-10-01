import torch
from torch import nn


class PNTrainingSigmoid(nn.Module):

    def __init__(self):
        super(PNTrainingSigmoid, self).__init__()
        return

    def forward(self, output_p, output_n, prior):
        cost = prior * torch.mean(torch.sigmoid(-output_p))
        cost = cost + (1 - prior) * torch.mean(torch.sigmoid(output_n))
        return cost


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
