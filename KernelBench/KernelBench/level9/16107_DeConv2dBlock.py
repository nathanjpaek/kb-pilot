import torch
from torch import nn


class DeConv2dBlock(nn.Module):
    """
    Similar to a LeNet block
    4x upsampling, dimension hard-coded
    """

    def __init__(self, in_dim: 'int', hidden_dim: 'int', out_dim: 'int',
        stride: 'int'=2, kernel_size: 'int'=3, padding: 'int'=2,
        output_padding: 'int'=1, dropout=0.1, activation_type='silu', debug
        =False):
        super(DeConv2dBlock, self).__init__()
        padding1 = padding // 2 if padding // 2 >= 1 else 1
        self.deconv0 = nn.ConvTranspose2d(in_channels=in_dim, out_channels=
            hidden_dim, kernel_size=kernel_size, stride=stride,
            output_padding=output_padding, padding=padding)
        self.deconv1 = nn.ConvTranspose2d(in_channels=hidden_dim,
            out_channels=out_dim, kernel_size=kernel_size, stride=stride,
            output_padding=output_padding, padding=padding1)
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.debug = debug

    def forward(self, x):
        x = self.deconv0(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.deconv1(x)
        x = self.activation(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'hidden_dim': 4, 'out_dim': 4}]
