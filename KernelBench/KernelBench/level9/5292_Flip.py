import torch
import torch.nn as nn
import torch.nn.init


class Flip(nn.Module):
    """Does horizontal or vertical flip on a BCHW tensor.

    Args:
        horizontal (bool): If True, applies horizontal flip. Else, vertical
            flip is applied. Default = True

    ** Not recommended for CPU (Pillow/OpenCV based functions are faster).
    """

    def __init__(self, horizontal: 'bool'=True):
        super(Flip, self).__init__()
        self.horizontal = horizontal

    def forward(self, tensor: 'torch.Tensor'):
        return tensor.flip(3 if self.horizontal else 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
