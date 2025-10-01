import torch
import torch.nn as nn
import torch.utils.data
import torch
import torch.nn.parallel


class VNMaxPool(nn.Module):

    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        """
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        """
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]
            ) + (idx,)
        x_max = x[index_tuple]
        return x_max


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
