import torch
import torch as th
import torch.nn as nn


def minimum(x, dim=-1, scale_up=False, inplace=False):
    if inplace:
        x_ = x.clone()
        min_x = th.min(x_, dim=dim, keepdim=True)[0]
        min_mask = x_ == min_x
        x.masked_fill_(min_mask == 0, 0.0)
        if scale_up:
            x_sum = th.sum(x_, dim=dim, keepdim=True)
            min_sum = th.sum(x, dim=dim, keepdim=True)
            scale = x_sum / min_sum
            scale.masked_fill_(scale.isnan(), 0.0)
            x *= scale
        return x
    else:
        min_x = th.min(x, dim=dim, keepdim=True)[0]
        min_mask = x == min_x
        masked_x = x * min_mask.float()
        if scale_up:
            x_sum = th.sum(x, dim=dim, keepdim=True)
            min_sum = th.sum(masked_x, dim=dim, keepdim=True)
            scale = x_sum / min_sum
            scale.masked_fill_(scale.isnan(), 0.0)
            masked_x = masked_x * scale
        return masked_x


class Minimum(nn.Module):

    def __init__(self, dim=-1, scale_up=False, inplace=False):
        super(Minimum, self).__init__()
        self.dim = dim
        self.scale_up = scale_up
        self.inplace = inplace

    def forward(self, x):
        return minimum(x, self.dim, self.scale_up, self.inplace)

    def extra_repr(self):
        return 'dim={}, scale_up={}{}'.format(self.dim, self.scale_up, 
            ', inplace=True' if self.inplace else '')


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
