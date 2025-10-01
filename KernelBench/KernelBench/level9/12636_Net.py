import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, r0, c0):
        super(Net, self).__init__()
        self.r = nn.Parameter(torch.FloatTensor([r0]))
        self.c = nn.Parameter(torch.FloatTensor([c0]))

    def forward(self):
        cube_r = -3 * self.c * self.c * self.r + self.r * self.r * self.r
        cube_c = 3 * self.c * self.r * self.r - self.c * self.c * self.c
        fin_r = cube_r - 1
        fin_c = cube_c
        return fin_r, fin_c


def get_inputs():
    return []


def get_init_inputs():
    return [[], {'r0': 4, 'c0': 4}]
