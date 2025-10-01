from torch.nn import Module
import torch
import torch.nn.functional as F
import torch.nn
from torch.nn.modules.module import Module


class AffineGridGen(Module):

    def __init__(self, out_h=240, out_w=240, out_ch=3, use_cuda=True):
        super(AffineGridGen, self).__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch

    def forward(self, theta):
        b = theta.size()[0]
        if not theta.size() == (b, 2, 3):
            theta = theta.view(-1, 2, 3)
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size, self.out_ch, self.out_h, self.out_w)
            )
        return F.affine_grid(theta, out_size)


def get_inputs():
    return [torch.rand([4, 2, 3])]


def get_init_inputs():
    return [[], {}]
