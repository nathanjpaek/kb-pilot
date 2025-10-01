import torch
import torch.nn.functional


class CatModel(torch.nn.Module):

    def __init__(self):
        super(CatModel, self).__init__()

    def forward(self, x):
        return torch.cat([x, x])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
