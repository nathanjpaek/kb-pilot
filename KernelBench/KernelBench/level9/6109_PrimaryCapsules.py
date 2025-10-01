import torch
import torch.nn as nn


def squash(x, dim=-1, epsilon=1e-08):
    norm = (x ** 2).sum(dim=dim, keepdim=True)
    x = norm / (norm + 1) * x / (torch.sqrt(norm) + epsilon)
    return x


class PrimaryCapsules(nn.Module):

    def __init__(self, in_features, capsules_num, capsules_dim):
        super(PrimaryCapsules, self).__init__()
        self.in_features = in_features
        self.capsules_num = capsules_num
        self.capsules_dim = capsules_dim
        self.conv = nn.Conv2d(in_features, capsules_num * capsules_dim,
            kernel_size=9, stride=2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        """
        Since all capsules use the same convolution operations, just do once and reshape.
        """
        x = x.view(batch_size, -1, self.capsules_dim)
        x = squash(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'in_features': 4, 'capsules_num': 4, 'capsules_dim': 4}]
