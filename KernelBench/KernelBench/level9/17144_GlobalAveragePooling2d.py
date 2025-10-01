import torch
import torch as pt
import torch.nn as nn


class GlobalAveragePooling2d(nn.Module):
    """class for performing global average pooling on 2d feature maps"""

    def forward(self, x):
        """
        calculates the average of each feature map in the tensor

        :param x: input tensor of shape [batch, channels, height, width]
        :return: tensor that containes the average for each channel, [batch, channel_average]
        """
        return pt.mean(pt.mean(x, -1), -1)[..., None, None]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
