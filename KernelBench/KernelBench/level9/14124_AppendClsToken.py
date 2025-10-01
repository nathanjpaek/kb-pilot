import torch
import torch.nn as nn
from functools import partial
import torch.utils.cpp_extension


class AppendClsToken(nn.Module):

    def __init__(self, embed_dim, init_func=partial(nn.init.normal_, std=0.02)
        ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        init_func(self.cls_token)

    def forward(self, x):
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)
        return torch.cat([x, cls_token], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'embed_dim': 4}]
