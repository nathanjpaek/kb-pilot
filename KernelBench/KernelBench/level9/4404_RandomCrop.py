import torch
from torch import nn


def choose_rand_patches(x, patch_sz, dim):
    assert dim == 2 or dim == 3
    batch_sz = x.shape[0]
    patches = x.unfold(dim, patch_sz, 1)
    n_patches = patches.shape[2]
    idx = torch.randint(0, n_patches, (batch_sz,))
    if dim == 2:
        patches = patches[torch.arange(batch_sz), :, idx, :]
    elif dim == 3:
        patches = patches[torch.arange(batch_sz), :, :, idx]
    return patches


class RandomCrop(nn.Module):

    def __init__(self, crop_sz):
        super(RandomCrop, self).__init__()
        self.crop_sz = crop_sz

    def forward(self, x):
        img_sz = x.shape[-1]
        assert img_sz >= self.crop_sz, f'img_sz {img_sz} is too small for crop_sz {self.crop_sz}'
        x = choose_rand_patches(x, self.crop_sz, 2)
        x = choose_rand_patches(x, self.crop_sz, 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'crop_sz': 4}]
