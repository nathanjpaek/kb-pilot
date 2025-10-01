from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class segmentation_layer(nn.Module):

    def __init__(self, args):
        super(segmentation_layer, self).__init__()
        self.segm_layer = nn.Conv2d(32, args.snumclass, kernel_size=1)

    def forward(self, featMap):
        segm = self.segm_layer(featMap)
        return segm


def get_inputs():
    return [torch.rand([4, 32, 64, 64])]


def get_init_inputs():
    return [[], {'args': _mock_config(snumclass=4)}]
