import torch
import torch.nn as nn
import torch.nn


class OrModule(nn.Module):

    def forward(self, attn1, attn2):
        out = torch.max(attn1, attn2)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
