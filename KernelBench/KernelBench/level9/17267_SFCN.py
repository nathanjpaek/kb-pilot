import torch
import torch.nn as nn


class SFCN(nn.Module):

    def __init__(self):
        super(SFCN, self).__init__()
        cnn = nn.Sequential()
        input_c = [3, 18, 18]
        padding = [3, 3, 6]
        dilation = [1, 1, 2]
        for i in range(3):
            cnn.add_module('sfcn{}'.format(i), nn.Conv2d(input_c[i], 18, 7,
                padding=padding[i], dilation=dilation[i]))
            cnn.add_module('sfcn_relu{}'.format(i), nn.ReLU(True))
        self.cnn = cnn

    def forward(self, input):
        output = self.cnn(input)
        return output


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
