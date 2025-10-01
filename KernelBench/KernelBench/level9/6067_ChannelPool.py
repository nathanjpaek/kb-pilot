import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelPool(nn.MaxPool1d):

    def forward(self, X):
        X = X.permute(1, 2, 0)
        pooled = F.max_pool1d(X, self.kernel_size)
        pooled = pooled.permute(2, 0, 1).squeeze(0)
        return pooled


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
