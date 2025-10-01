from torch.nn import Module
import torch
import torch.nn as nn
import torch.utils.data


def patchify2d(w_in, w_out, k, *, bias=True):
    """Helper for building a patchify layer as used by ViT models."""
    return nn.Conv2d(w_in, w_out, k, stride=k, padding=0, bias=bias)


def patchify2d_cx(cx, w_in, w_out, k, *, bias=True):
    """Accumulates complexity of patchify2d into cx = (h, w, flops, params, acts)."""
    err_str = 'Only kernel sizes divisible by the input size are supported.'
    assert cx['h'] % k == 0 and cx['w'] % k == 0, err_str
    h, w, flops, params, acts = cx['h'], cx['w'], cx['flops'], cx['params'
        ], cx['acts']
    h, w = h // k, w // k
    flops += k * k * w_in * w_out * h * w + (w_out * h * w if bias else 0)
    params += k * k * w_in * w_out + (w_out if bias else 0)
    acts += w_out * h * w
    return {'h': h, 'w': w, 'flops': flops, 'params': params, 'acts': acts}


class ViTStemPatchify(Module):
    """The patchify vision transformer stem as per https://arxiv.org/abs/2010.11929."""

    def __init__(self, w_in, w_out, k):
        super(ViTStemPatchify, self).__init__()
        self.patchify = patchify2d(w_in, w_out, k, bias=True)

    def forward(self, x):
        return self.patchify(x)

    @staticmethod
    def complexity(cx, w_in, w_out, k):
        return patchify2d_cx(cx, w_in, w_out, k, bias=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'w_in': 4, 'w_out': 4, 'k': 4}]
