import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F


class Sobel(nn.Module):

    def __init__(self):
        super(Sobel, self).__init__()
        kernel = [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-1, -2, -1], [0, 
            0, 0], [1, 2, 1]]]
        kernel = torch.Tensor(kernel).unsqueeze(1).repeat([3, 1, 1, 1])
        self.kernel = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            It represents a RGB image with pixel values in [0, 1] range.
        Returns:
            a float tensor with shape [b, 3*2, h, w].
        """
        x = F.conv2d(x, self.kernel, padding=1, groups=3)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
