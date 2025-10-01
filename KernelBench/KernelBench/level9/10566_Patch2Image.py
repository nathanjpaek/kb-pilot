import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class Patch2Image(nn.Module):
    """ take in patch and copy n_up times to form the full image"""

    def __init__(self, patch_sz, n_up):
        super(Patch2Image, self).__init__()
        self.patch_sz = patch_sz
        self.n_up = n_up

    def forward(self, x):
        assert x.shape[-1
            ] == self.patch_sz, f'inp.patch_sz ({x.shape[-1]}): =/= self.patch_sz ({self.patch_sz})'
        x = torch.cat([x] * self.n_up, -1)
        x = torch.cat([x] * self.n_up, -2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'patch_sz': 4, 'n_up': 4}]
