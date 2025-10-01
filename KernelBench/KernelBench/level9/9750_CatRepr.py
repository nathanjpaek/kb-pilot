import torch
import torch.nn as nn


class CatRepr(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, data_list):
        cat_regions = [torch.cat([hidden[0], torch.mean(hidden, dim=0),
            hidden[-1]], dim=-1).view(1, -1) for hidden in data_list]
        cat_out = torch.cat(cat_regions, dim=0)
        return cat_out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
