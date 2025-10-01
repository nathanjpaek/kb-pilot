import torch
import torch.utils.data
import torch.nn as nn
import torch.utils.checkpoint


class AvgSpacial(nn.Module):

    def forward(self, inp):
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
