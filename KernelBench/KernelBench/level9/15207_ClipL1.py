import torch
import torch.nn as nn


class ClipL1(nn.Module):
    """ Clip L1 loss
    From: https://github.com/HolmesShuan/AIM2020-Real-Super-Resolution/
    ClipL1 Loss combines Clip function and L1 loss. self.clip_min sets the
    gradients of well-trained pixels to zeros and clip_max works as a noise filter.
    data range [0, 255]: (clip_min=0.0, clip_max=10.0),
    for [0,1] set clip_min to 1/255=0.003921.
    """

    def __init__(self, clip_min=0.0, clip_max=10.0):
        super(ClipL1, self).__init__()
        self.clip_max = clip_max
        self.clip_min = clip_min

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        loss = torch.mean(torch.clamp(torch.abs(x - y), self.clip_min, self
            .clip_max))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
