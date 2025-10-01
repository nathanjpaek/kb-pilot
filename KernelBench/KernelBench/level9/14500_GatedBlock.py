import torch
import torch.nn as nn


class GatedBlock(nn.Module):

    def __init__(self, dilation: 'int', w_dim: 'int'):
        """Gated block with sigmoid/tanh gates."""
        super().__init__()
        self.dilation = dilation
        self.tanh_conv = nn.Conv2d(w_dim, w_dim, kernel_size=(2, 1),
            dilation=(dilation, 1), groups=w_dim)
        self.sigmoid_conv = nn.Conv2d(w_dim, w_dim, kernel_size=(2, 1),
            dilation=(dilation, 1), groups=w_dim)
        self.out_conv = nn.Conv2d(w_dim, w_dim, kernel_size=1, groups=w_dim)

    def forward(self, x_in):
        x_tanh, x_sigmoid = self.tanh_conv(x_in), self.sigmoid_conv(x_in)
        x_gate = torch.tanh(x_tanh) * torch.sigmoid(x_sigmoid)
        x_out = self.out_conv(x_gate + x_in[:, :, :x_gate.size(2), :])
        return x_out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dilation': 1, 'w_dim': 4}]
