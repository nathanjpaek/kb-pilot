import torch
import torch.nn as nn


class CE(nn.Module):

    def __init__(self):
        super(CE, self).__init__()

    def forward(self, mat1, mat2):
        return -torch.mean(mat2 * torch.log(mat1 + 1e-10) + (1 - mat2) *
            torch.log(1 - mat1 + 1e-10))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
