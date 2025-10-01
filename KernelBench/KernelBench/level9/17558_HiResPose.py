import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple
import torch.nn.functional as F


class HiResPose(nn.Module):
    """
    GNINA HiResPose model architecture.

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)

    Notes
    -----
    This architecture was translated from the following Caffe model:

        https://github.com/gnina/models/blob/master/crossdocked_paper/hires_pose.model

    The main difference is that the PyTorch implementation resurns the log softmax.

    This model is implemented only for multi-task pose and affinity prediction.
    """

    def __init__(self, input_dims: 'Tuple'):
        super().__init__()
        self.input_dims = input_dims
        self.features = nn.Sequential(OrderedDict([('unit1_conv', nn.Conv3d
            (in_channels=input_dims[0], out_channels=32, kernel_size=3,
            stride=1, padding=1)), ('unit1_func', nn.ReLU()), ('unit2_pool',
            nn.MaxPool3d(kernel_size=2, stride=2)), ('unit2_conv', nn.
            Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1,
            padding=1)), ('unit2_func', nn.ReLU()), ('unit3_pool', nn.
            MaxPool3d(kernel_size=2, stride=2)), ('unit3_conv', nn.Conv3d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1,
            padding=1)), ('unit3_func', nn.ReLU())]))
        self.features_out_size = input_dims[1] // 4 * input_dims[2
            ] // 4 * input_dims[3] // 4 * 128
        self.pose = nn.Sequential(OrderedDict([('pose_output', nn.Linear(
            in_features=self.features_out_size, out_features=2))]))
        self.affinity = nn.Sequential(OrderedDict([('affinity_output', nn.
            Linear(in_features=self.features_out_size, out_features=1))]))

    def forward(self, x: 'torch.Tensor'):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Notes
        -----
        The pose score is the log softmax of the output of the last linear layer.
        """
        x = self.features(x)
        None
        x = x.view(-1, self.features_out_size)
        pose_raw = self.pose(x)
        pose_log = F.log_softmax(pose_raw, dim=1)
        affinity = self.affinity(x)
        return pose_log, affinity.squeeze(-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dims': [4, 4, 4, 4]}]
