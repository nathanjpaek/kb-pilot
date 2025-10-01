import torch
import torch.nn as nn


class NormAttnMap(nn.Module):

    def __init__(self, norm_type='cossim'):
        super(NormAttnMap, self).__init__()
        self.norm_type = norm_type

    def forward(self, attn_map):
        if self.norm_type != 'cosssim':
            norm = torch.max(attn_map, dim=1, keepdim=True)[0].detach()
        else:
            norm = torch.max(torch.abs(attn_map), dim=1, keepdim=True)[0
                ].detach()
        norm[norm <= 1] = 1
        attn = attn_map / norm
        return attn, norm


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
