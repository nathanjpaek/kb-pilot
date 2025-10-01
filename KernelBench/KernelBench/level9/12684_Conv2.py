import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn import Conv3d


class Conv2(nn.Module):

    def __init__(self):
        super(Conv2, self).__init__()
        self.conv1 = Conv2d(in_channels=10, out_channels=2, kernel_size=5,
            padding=2, bias=True)
        self.conv2 = Conv3d(in_channels=2, out_channels=10, kernel_size=5,
            padding=2, bias=True)

    def forward(self, x):
        grey_x = self.conv1(x)
        grey_xx = torch.stack([grey_x[:, 0, :, :]] + 9 * [grey_x[:, 1, :, :
            ]], dim=1)
        assert grey_xx.shape[1] == 10
        stack_x = torch.stack([x, x - grey_xx], dim=1)
        return self.conv2(stack_x)


def get_inputs():
    return [torch.rand([4, 10, 64, 64])]


def get_init_inputs():
    return [[], {}]
