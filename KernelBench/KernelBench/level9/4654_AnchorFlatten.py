import torch
import torch.nn as nn
import torch.utils.data


class AnchorFlatten(nn.Module):
    """
        Module for anchor-based network outputs,
        Init args:
            num_output: number of output channel for each anchor.

        Forward args:
            x: torch.tensor of shape [B, num_anchors * output_channel, H, W]

        Forward return:
            x : torch.tensor of shape [B, num_anchors * H * W, output_channel]
    """

    def __init__(self, num_output_channel):
        super(AnchorFlatten, self).__init__()
        self.num_output_channel = num_output_channel

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(x.shape[0], -1, self.num_output_channel)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_output_channel': 4}]
