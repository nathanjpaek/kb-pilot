import torch
from torch import nn


def depth_to_3d(depth: 'torch.Tensor', xyz: 'torch.Tensor') ->torch.Tensor:
    points_depth: 'torch.Tensor' = depth.permute(0, 2, 3, 1)
    points_3d: 'torch.Tensor' = xyz * points_depth
    return points_3d.permute(0, 3, 1, 2)


class Model(nn.Module):

    def forward(self, xyz, depth):
        depthFP16 = 256.0 * depth[:, :, :, 1::2] + depth[:, :, :, ::2]
        return depth_to_3d(depthFP16, xyz)


def get_inputs():
    return [torch.rand([4, 4, 2, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
