import torch
import torch.optim
import torch.nn as nn


class FeatureSelect(nn.Module):

    def __init__(self, in_dim=84, ratio=0.5):
        """
        Feature Select via Sorting

        Args:
            in_dim: the number of dimensions of raw features
            ratio: the portion of selected features
        """
        super(FeatureSelect, self).__init__()
        self.in_dim = in_dim
        self.select_dim = int(in_dim * ratio)

    def forward(self, x):
        """
        Args:
            x: feature discrepancy of shape [batch_size, in_dim]
        Returns:
            v: selecting vector of shape [batch_size, in_dim]
        """
        idx = torch.argsort(x, dim=1)
        idx[idx < self.select_dim] = 1
        idx[idx >= self.select_dim] = 0
        v = idx
        return v


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
