import torch
import torch.nn as nn
import torch.nn.parallel


class RepeatChannel(nn.Module):

    def __init__(self, repeat):
        super(RepeatChannel, self).__init__()
        self.repeat = repeat

    def forward(self, img):
        return img.repeat(1, self.repeat, 1, 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'repeat': 4}]
