import torch
import numpy as np
import torch.nn as nn


class YOLOLayer(nn.Module):
    """
    Detection Layer
    """

    def __init__(self, in_ch, n_anchors, n_classes):
        super(YOLOLayer, self).__init__()
        self.n_anchors = n_anchors
        self.n_classes = n_classes
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=self.
            n_anchors * (self.n_classes + 5), kernel_size=1, stride=1,
            padding=0)

    def forward(self, x, targets=None):
        output = self.conv(x)
        batchsize = output.shape[0]
        fsize = output.shape[2]
        n_ch = 5 + self.n_classes
        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2)
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2,
            4:n_ch]])
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'n_anchors': 4, 'n_classes': 4}]
