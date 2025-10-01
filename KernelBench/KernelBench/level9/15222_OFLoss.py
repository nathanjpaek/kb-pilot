import torch
import torch.nn as nn


def get_outnorm(x: 'torch.Tensor', out_norm: 'str'='') ->torch.Tensor:
    """ Common function to get a loss normalization value. Can
        normalize by either the batch size ('b'), the number of
        channels ('c'), the image size ('i') or combinations
        ('bi', 'bci', etc)
    """
    img_shape = x.shape
    if not out_norm:
        return 1
    norm = 1
    if 'b' in out_norm:
        norm /= img_shape[0]
    if 'c' in out_norm:
        norm /= img_shape[-3]
    if 'i' in out_norm:
        norm /= img_shape[-1] * img_shape[-2]
    return norm


class OFLoss(nn.Module):
    """ Overflow loss (similar to Range limiting loss, needs tests)
    Penalizes for pixel values that exceed the valid range (default [0,1]).
    Note: This solves part of the SPL brightness problem and can be useful
    in other cases as well)
    """

    def __init__(self, legit_range=None, out_norm: 'str'='bci'):
        super(OFLoss, self).__init__()
        if legit_range is None:
            legit_range = [0, 1]
        self.legit_range = legit_range
        self.out_norm = out_norm

    def forward(self, x):
        norm = get_outnorm(x, self.out_norm)
        img_clamp = x.clamp(self.legit_range[0], self.legit_range[1])
        return torch.log((x - img_clamp).abs() + 1).sum() * norm


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
