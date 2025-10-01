import torch
import torch.nn as nn


class UnStackDelta(nn.Module):
    """Reverse of StackDelta"""

    def __init__(self):
        super().__init__()

    def forward(self, x: 'torch.Tensor'):
        assert x.dim() == 4
        if x.requires_grad:
            out = x.transpose(1, 2).contiguous()
        else:
            out = x.transpose_(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), out.size(2) * out.size(3))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
