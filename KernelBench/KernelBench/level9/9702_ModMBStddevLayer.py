import torch
import torch.nn as nn


class ModMBStddevLayer(nn.Module):
    """Modified MiniBatch Stddev Layer.

    This layer is modified from ``MiniBatchStddevLayer`` used in PGGAN. In
    StyleGAN2, the authors add a new feature, `channel_groups`, into this
    layer.
    """

    def __init__(self, group_size=4, channel_groups=1, sync_groups=None,
        eps=1e-08):
        super(ModMBStddevLayer, self).__init__()
        self.group_size = group_size
        self.eps = eps
        self.channel_groups = channel_groups
        self.sync_groups = group_size if sync_groups is None else sync_groups

    def forward(self, x):
        assert x.shape[0] <= self.group_size or x.shape[0
            ] % self.group_size == 0, f'Batch size be smaller than or equal to group size. Otherwise, batch size should be divisible by the group size.But got batch size {x.shape[0]}, group size {self.group_size}'
        assert x.shape[1
            ] % self.channel_groups == 0, f'"channel_groups" must be divided by the feature channels. channel_groups: {self.channel_groups}, feature channels: {x.shape[1]}'
        n, c, h, w = x.shape
        group_size = min(n, self.group_size)
        y = torch.reshape(x, (group_size, -1, self.channel_groups, c //
            self.channel_groups, h, w))
        y = torch.var(y, dim=0, unbiased=False)
        y = torch.sqrt(y + self.eps)
        y = y.mean(dim=(2, 3, 4), keepdim=True).squeeze(2)
        y = y.repeat(group_size, 1, h, w)
        return torch.cat([x, y], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
