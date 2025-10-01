import math
import torch
import torch.nn.functional as F
import torch.nn as nn


def cal_width_dim_2d(input_dim, kernel_size, stride, padding=1):
    return math.floor((input_dim + 2 * padding - kernel_size) / stride + 1)


class Conv2dLayer(nn.Module):

    def __init__(self, input_size, in_channel, out_channel, kernel_size,
        stride, dropout=0.1, batch_norm=False, residual=False,
        act_func_type='relu'):
        super(Conv2dLayer, self).__init__()
        self.input_size = input_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.batch_norm = batch_norm
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = 0, kernel_size // 2 if isinstance(self.kernel_size, int
            ) else kernel_size[1] // 2
        self.residual = residual
        self.act_func_type = act_func_type
        self.conv_layer = nn.Conv2d(in_channels=in_channel, out_channels=
            out_channel, kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding)
        self.output_size = cal_width_dim_2d(input_size, self.kernel_size if
            isinstance(self.kernel_size, int) else self.kernel_size[1], 
            self.stride if isinstance(self.stride, int) else self.stride[1],
            padding=self.padding if isinstance(self.padding, int) else self
            .padding[1])
        if self.batch_norm:
            self.norm = nn.BatchNorm2d(out_channel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """Forward computation.

        Args:
            x (FloatTensor): `[B, C_i, T, F]`
            mask (IntTensor): `[B, 1, T]`
        Returns:
            out (FloatTensor): `[B, C_o, T', F']`
            out_mask (IntTensor): `[B, 1, T]`

        """
        residual = x
        out = self.conv_layer(x)
        out = F.relu(out)
        if self.batch_norm:
            out = self.norm(out)
        out = self.dropout(out)
        if self.residual and out.size() == residual.size():
            out += residual
        mask = self.return_output_mask(mask, out.size(2))
        return out, mask

    def return_output_mask(self, mask, t):
        stride = self.stride if isinstance(self.stride, int) else self.stride[0
            ]
        kernel_size = self.kernel_size if isinstance(self.kernel_size, int
            ) else self.kernel_size[0]
        mask = mask[:, math.floor(kernel_size / 2)::stride][:, :t]
        return mask


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'in_channel': 4, 'out_channel': 4,
        'kernel_size': 4, 'stride': 1}]
