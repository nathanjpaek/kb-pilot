import torch
import torch.nn as nn


class Centered_Grad(nn.Module):

    def __init__(self):
        super(Centered_Grad, self).__init__()
        self.x_ker_init = torch.tensor([[[[-0.5, 0, 0.5]]]], dtype=torch.
            float, requires_grad=True)
        self.y_ker_init = torch.tensor([[[[-0.5], [0], [0.5]]]], dtype=
            torch.float, requires_grad=True)
        self.center_conv_x = nn.Conv2d(1, 1, (1, 3), padding=(0, 1), bias=False
            )
        self.center_conv_x.weight.data = self.x_ker_init
        self.center_conv_y = nn.Conv2d(1, 1, (3, 1), padding=(1, 0), bias=False
            )
        self.center_conv_y.weight.data = self.y_ker_init

    def forward(self, x):
        assert len(x.shape) == 4
        diff_x = self.center_conv_x(x)
        diff_y = self.center_conv_y(x)
        first_col = 0.5 * (x[:, :, :, 1:2] - x[:, :, :, 0:1])
        last_col = 0.5 * (x[:, :, :, -1:] - x[:, :, :, -2:-1])
        diff_x_valid = diff_x[:, :, :, 1:-1]
        diff_x = torch.cat((first_col, diff_x_valid, last_col), 3)
        first_row = 0.5 * (x[:, :, 1:2, :] - x[:, :, 0:1, :])
        last_row = 0.5 * (x[:, :, -1:, :] - x[:, :, -2:-1, :])
        diff_y_valid = diff_y[:, :, 1:-1, :]
        diff_y = torch.cat((first_row, diff_y_valid, last_row), 2)
        return diff_x, diff_y


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
