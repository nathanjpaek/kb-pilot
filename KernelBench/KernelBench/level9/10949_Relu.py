from torch.nn import Module
import torch


class Relu(Module):

    def forward(self, inp):
        return inp.clamp_min(0.0) - 0.5

    def bwd(self, out, inp):
        inp.g = (inp > 0).float() * out.g


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
