import torch
from torch import nn
import torch as th
import torch.hub
import torch.utils.data


class SelfGating(nn.Module):

    def __init__(self, input_dim):
        super(SelfGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, input_tensor):
        """Feature gating as used in S3D-G.
      """
        spatiotemporal_average = th.mean(input_tensor, dim=[2, 3, 4])
        weights = self.fc(spatiotemporal_average)
        weights = th.sigmoid(weights)
        return weights[:, :, None, None, None] * input_tensor


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
