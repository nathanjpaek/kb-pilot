import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalLoss(nn.Module):
    """ Variational loss to enforce continuity of images
    """

    def forward(self, input):
        """ forward pass """
        self.loss = F.mse_loss(input[:, :, 1:, :], input[:, :, :-1, :]
            ) + F.mse_loss(input[:, :, :, 1:], input[:, :, :, :-1])
        return input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
