import torch
import torch.nn as nn


class Normalization(nn.Module):

    def __init__(self):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(-
            1, 1, 1))
        self.std = nn.Parameter(torch.tensor([0.329, 0.224, 0.225]).view(-1,
            1, 1))

    def forward(self, img):
        return (img - self.mean) / self.std


def get_inputs():
    return [torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {}]
