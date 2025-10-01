import torch
import torch.utils.data
import torch.nn as nn
import torch.utils.checkpoint


class Merge(nn.Module):

    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
