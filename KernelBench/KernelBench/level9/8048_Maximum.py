import torch
import torch as th
import torch.nn as nn


def maximum(x, dim=-1, scale_up=False, inplace=False):
    if inplace:
        x_ = x.clone()
        max_x = th.max(x_, dim=dim, keepdim=True)[0]
        max_mask = x_ == max_x
        x.masked_fill_(max_mask == 0, 0.0)
        if scale_up:
            x_sum = th.sum(x_, dim=dim, keepdim=True)
            max_sum = th.sum(x, dim=dim, keepdim=True)
            scale = x_sum / max_sum
            scale.masked_fill_(scale.isnan(), 0.0)
            x *= scale
        return x
    else:
        max_x = th.max(x, dim=dim, keepdim=True)[0]
        max_mask = x == max_x
        masked_x = x * max_mask.float()
        if scale_up:
            x_sum = th.sum(x, dim=dim, keepdim=True)
            max_sum = th.sum(masked_x, dim=dim, keepdim=True)
            scale = x_sum / max_sum
            scale.masked_fill_(scale.isnan(), 0.0)
            masked_x = masked_x * scale
        return masked_x


class Maximum(nn.Module):

    def __init__(self, dim=-1, scale_up=False, inplace=False):
        super(Maximum, self).__init__()
        self.dim = dim
        self.scale_up = scale_up
        self.inplace = inplace

    def forward(self, x):
        return maximum(x, self.dim, self.scale_up, self.inplace)

    def extra_repr(self):
        return 'dim={}, scale_up={}{}'.format(self.dim, self.scale_up, 
            ', inplace=True' if self.inplace else '')


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
