import torch
import torch.nn as nn


class Decoder1(nn.Module):

    def __init__(self):
        super(Decoder1, self).__init__()
        self.reflecPad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x):
        out = self.reflecPad2(x)
        out = self.conv3(out)
        return out


def get_inputs():
    return [torch.rand([4, 64, 4, 4])]


def get_init_inputs():
    return [[], {}]
