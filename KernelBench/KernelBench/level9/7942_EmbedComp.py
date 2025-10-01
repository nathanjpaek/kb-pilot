import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn


class EmbedComp(nn.Module):

    def __init__(self, insize, outsize, md):
        super().__init__()
        self.fc1 = nn.Linear(insize, outsize)
        self.outsize = outsize
        self.md = md

    def forward(self, x):
        out = self.fc1(x)
        out = out.view(-1, self.outsize, 1, 1)
        out = out.repeat(1, 1, self.md, self.md)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'insize': 4, 'outsize': 4, 'md': 4}]
