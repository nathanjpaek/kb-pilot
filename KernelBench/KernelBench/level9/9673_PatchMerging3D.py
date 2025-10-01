import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel


class PatchMerging3D(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, isotropy=False):
        super().__init__()
        self.dim = dim
        self.isotropy = isotropy
        if self.isotropy:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        else:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        _B, _D, H, W, _C = x.shape
        pad_input = H % 2 == 1 or W % 2 == 1
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        if self.isotropy:
            x0 = x[:, 0::2, 0::2, 0::2, :]
            x1 = x[:, 0::2, 1::2, 0::2, :]
            x2 = x[:, 0::2, 0::2, 1::2, :]
            x3 = x[:, 0::2, 1::2, 1::2, :]
            x4 = x[:, 1::2, 0::2, 0::2, :]
            x5 = x[:, 1::2, 1::2, 0::2, :]
            x6 = x[:, 1::2, 0::2, 1::2, :]
            x7 = x[:, 1::2, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        else:
            x0 = x[:, :, 0::2, 0::2, :]
            x1 = x[:, :, 1::2, 0::2, :]
            x2 = x[:, :, 0::2, 1::2, :]
            x3 = x[:, :, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
