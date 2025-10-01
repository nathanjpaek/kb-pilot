import torch
import torch.nn as nn


class L2Norm(nn.Module):
    """
       Scale shall be learnable according to original paper
       scale: initial scale number
       chan_num: L2Norm channel number (norm over all channels)
    """

    def __init__(self, scale=20, chan_num=512):
        super(L2Norm, self).__init__()
        self.scale = nn.Parameter(torch.Tensor([scale] * chan_num).view(1,
            chan_num, 1, 1))

    def forward(self, data):
        return self.scale * data * data.pow(2).sum(dim=1, keepdim=True).clamp(
            min=1e-12).rsqrt()


def get_inputs():
    return [torch.rand([4, 512, 4, 4])]


def get_init_inputs():
    return [[], {}]
