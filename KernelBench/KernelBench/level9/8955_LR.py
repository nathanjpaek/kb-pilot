import torch
import torch.nn as nn
import torch.utils.data


class LR(nn.Module):

    def __init__(self, feature_nums, output_dim=1):
        super(LR, self).__init__()
        self.linear = nn.Linear(feature_nums, output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: pctrs
        """
        out = self.bias + torch.sum(self.linear(x), dim=1)
        return out.unsqueeze(1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_nums': 4}]
