import torch
import torch.nn as nn


class GroupPointWise(nn.Module):

    def __init__(self, in_dim, n_heads=4, proj_factor=1, target_dim=None):
        super().__init__()
        if target_dim is not None:
            proj_ch = target_dim // proj_factor
        else:
            proj_ch = in_dim // proj_factor
        self.w = nn.Parameter(torch.Tensor(in_dim, n_heads, proj_ch // n_heads)
            )
        nn.init.normal_(self.w, std=0.01)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out = torch.einsum('bhwc,cnp->bnhwp', x, self.w)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4}]
