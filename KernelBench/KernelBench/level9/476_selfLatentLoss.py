import torch
import torch.nn as nn


class selfLatentLoss(nn.Module):

    def __init__(self):
        super(selfLatentLoss, self).__init__()

    def forward(self, z_mean, z_log_sigma_sq):
        return torch.mean(torch.sum(torch.pow(z_mean, 2) + torch.exp(
            z_log_sigma_sq) - z_log_sigma_sq - 1, 1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
