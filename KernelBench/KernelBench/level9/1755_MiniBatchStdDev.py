import torch
import torch.nn as nn


class MiniBatchStdDev(nn.Module):
    """Layer that appends to every element of a batch a new ftr map containing the std of its group."""

    def __init__(self, group_sz=4, unbiased_std=False):
        super().__init__()
        self.group_sz = group_sz
        self.unbiased_std = unbiased_std

    def forward(self, x):
        bs, n_ch, h, w = x.shape
        x_groups = x.view(-1, self.group_sz, n_ch, h, w)
        stds_by_chw = x_groups.std(dim=1, unbiased=self.unbiased_std)
        mean_std = stds_by_chw.mean(dim=[1, 2, 3], keepdim=True)
        new_ftr_map = mean_std.unsqueeze(-1).repeat(1, self.group_sz, 1, h, w
            ).view(bs, 1, h, w)
        return torch.cat([x, new_ftr_map], axis=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
