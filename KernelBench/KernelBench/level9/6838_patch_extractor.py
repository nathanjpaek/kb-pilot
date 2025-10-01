import torch
from torch import nn


class patch_extractor(nn.Module):
    """
    Module for creating custom patch extractor
    """

    def __init__(self, patch_size, pad=False, center=False, dim=2):
        super(patch_extractor, self).__init__()
        self.dim = dim
        self.im2pat = nn.Unfold(kernel_size=patch_size)
        self.pad = pad
        self.padsize = patch_size - 1
        self.center = center
        self.patch_size = patch_size

    def forward(self, input, batch_size=0, split=[1, 0]):
        if self.pad and self.dim == 2:
            input = torch.cat((input, input[:, :, :self.padsize, :]), 2)
            input = torch.cat((input, input[:, :, :, :self.padsize]), 3)
        elif self.pad and self.dim == 3:
            input = torch.cat((input, input[:, :, :self.padsize, :, :]), 2)
            input = torch.cat((input, input[:, :, :, :self.padsize, :]), 3)
            input = torch.cat((input, input[:, :, :, :, :self.padsize]), 4)
        if self.dim == 2:
            patches = self.im2pat(input).squeeze(0).transpose(1, 0)
            split_size = patches.size(0) // split[0]
            if split[1] == split[0] - 1:
                patches = patches[split_size * split[1]:]
            else:
                patches = patches[split_size * split[1]:split_size * (split
                    [1] + 1)]
        elif self.dim == 3:
            patches = self.im2pat(input[0]).squeeze(0).transpose(1, 0).reshape(
                -1, input.shape[2], self.patch_size, self.patch_size)
            split_size = patches.size(0) // split[0]
            if split[1] == split[0] - 1:
                patches = patches[split_size * split[1]:]
            else:
                patches = patches[split_size * split[1]:split_size * (split
                    [1] + 1)]
            patches = patches.unfold(1, self.patch_size, self.stride).permute(
                0, 1, 4, 2, 3)
            patches = patches.reshape(-1, self.patch_size ** 3)
        if batch_size > 0:
            idx = torch.randperm(patches.size(0))[:batch_size]
            patches = patches[idx, :]
        if self.center:
            patches = patches - torch.mean(patches, -1).unsqueeze(-1)
        return patches


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'patch_size': 4}]
