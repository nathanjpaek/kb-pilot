import torch
import torch.nn as nn


class complex_relu_layer(nn.Module):

    def __init__(self):
        super(complex_relu_layer, self).__init__()

    def complex_relu(self, real, img):
        mask = 1.0 * (real >= 0)
        return mask * real, mask * img

    def forward(self, real, img=None):
        if img is None:
            img = real[1]
            real = real[0]
        real, img = self.complex_relu(real, img)
        return real, img


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
