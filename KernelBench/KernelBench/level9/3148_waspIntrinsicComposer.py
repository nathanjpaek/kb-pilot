from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class waspIntrinsicComposer(nn.Module):

    def __init__(self, opt):
        super(waspIntrinsicComposer, self).__init__()
        self.ngpu = opt.ngpu
        self.nc = opt.nc

    def forward(self, shading, albedo):
        self.shading = shading.repeat(1, self.nc, 1, 1)
        self.img = torch.mul(self.shading, albedo)
        return self.img


def get_inputs():
    return [torch.rand([4, 16, 4, 4]), torch.rand([4, 64, 4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config(ngpu=False, nc=4)}]
