import torch
import torch.nn as nn


class TotalVariationLoss(nn.Module):

    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            It represents a RGB image with pixel values in [0, 1] range.
        Returns:
            a float tensor with shape [].
        """
        h, w = x.size()[2:]
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h - 1, :], 2)
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w - 1], 2)
        return h_tv.mean([0, 1, 2, 3]) + w_tv.mean([0, 1, 2, 3])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
