import torch
import torch.nn as nn


class patch_extractor(nn.Module):
    """
    Module for creating custom patch extractor
    """

    def __init__(self, patch_size, pad=False):
        super(patch_extractor, self).__init__()
        self.im2pat = nn.Unfold(kernel_size=patch_size)
        self.pad = pad
        self.padsize = patch_size - 1

    def forward(self, input, batch_size=0):
        if self.pad:
            input = torch.cat((input, input[:, :, :self.padsize, :]), 2)
            input = torch.cat((input, input[:, :, :, :self.padsize]), 3)
        patches = self.im2pat(input).squeeze(0).transpose(1, 0)
        if batch_size > 0:
            idx = torch.randperm(patches.size(0))[:batch_size]
            patches = patches[idx, :]
        return patches


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'patch_size': 4}]
