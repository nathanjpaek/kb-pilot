from torch.nn import Module
import torch
import torch.nn
import torch.utils.data


class ImgLayerNorm(Module):
    """
    LayerNorm for images with channel axis 1
    (this is necessary because PyTorch's LayerNorm operates on the last axis)
    """

    def __init__(self, in_dim, eps=1e-05):
        super().__init__()
        self.in_dim = in_dim
        self.layernorm = torch.nn.LayerNorm(in_dim, eps=eps)

    def forward(self, x):
        _B, C, _H, _W = x.shape
        assert C == self.in_dim
        out = self.layernorm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        assert out.shape == x.shape
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4}]
