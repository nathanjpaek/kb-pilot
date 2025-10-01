import torch
import torch.nn as nn
import torch.nn.functional as F


class Image2Patch(nn.Module):
    """Some Information about Image2Patch"""

    def __init__(self, channels, image_size, patch_size):
        super(Image2Patch, self).__init__()
        if type(patch_size) == int:
            patch_size = [patch_size, patch_size]
        self.patch_size = patch_size
        if type(image_size) == int:
            image_size = [image_size, image_size]
        self.image_size = image_size
        self.channels = channels
        self.num_patch = [image_size[0] // patch_size[0], image_size[1] //
            patch_size[1]]

    def forward(self, x):
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        x = x.swapaxes(1, 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'image_size': 4, 'patch_size': 4}]
