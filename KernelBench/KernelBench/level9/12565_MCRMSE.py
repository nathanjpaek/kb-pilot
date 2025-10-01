import torch
from torch import nn


class MCRMSE(nn.Module):

    def __init__(self, num_scored=3, eps=1e-08):
        super().__init__()
        self.mse = nn.MSELoss()
        self.num_scored = num_scored
        self.eps = eps

    def forward(self, outputs, targets):
        score = 0
        for idx in range(self.num_scored):
            score += torch.sqrt(self.mse(outputs[:, :, idx], targets[:, :,
                idx]) + self.eps) / self.num_scored
        return score


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
