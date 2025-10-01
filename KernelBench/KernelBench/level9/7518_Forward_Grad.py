import torch
import torch.nn as nn
import torch.nn.functional as F


class Forward_Grad(nn.Module):

    def __init__(self):
        super(Forward_Grad, self).__init__()
        self.x_ker_init = torch.tensor([[[[-1, 1]]]], dtype=torch.float,
            requires_grad=True)
        self.y_ker_init = torch.tensor([[[[-1], [1]]]], dtype=torch.float,
            requires_grad=True)
        self.forward_conv_x = nn.Conv2d(1, 1, (1, 2), bias=False)
        self.forward_conv_x.weight.data = self.x_ker_init
        self.forward_conv_y = nn.Conv2d(1, 1, (2, 1), bias=False)
        self.forward_conv_y.weight.data = self.y_ker_init

    def forward(self, x):
        assert len(x.shape) == 4
        x.padright = F.pad(x, (0, 1, 0, 0))
        x.padbottom = F.pad(x, (0, 0, 0, 1))
        diff_x = self.forward_conv_x(x.padright)
        diff_y = self.forward_conv_y(x.padbottom)
        diff_x[:, :, :, x.shape[3] - 1] = 0
        diff_y[:, :, x.shape[2] - 1, :] = 0
        return diff_x, diff_y


def get_inputs():
    return [torch.rand([4, 1, 4, 4])]


def get_init_inputs():
    return [[], {}]
