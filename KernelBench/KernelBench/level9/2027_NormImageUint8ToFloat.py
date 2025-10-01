from torch.nn import Module
import torch


class NormImageUint8ToFloat(Module):

    def forward(self, im):
        return 2.0 * (im / 255.0 - 0.5)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
