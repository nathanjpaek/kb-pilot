import torch
from torch import nn


class OutputTransition(nn.Module):
    """
    Decoder output layer
    output the prediction of segmentation result
    """

    def __init__(self, inChans, outChans):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans,
            kernel_size=1)
        self.actv1 = torch.sigmoid

    def forward(self, x):
        return self.actv1(self.conv1(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inChans': 4, 'outChans': 4}]
