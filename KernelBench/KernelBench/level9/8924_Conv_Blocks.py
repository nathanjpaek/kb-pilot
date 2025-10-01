import torch
import torch.nn as nn


class Conv_Blocks(nn.Module):

    def __init__(self, input_dim, output_dim, filter_size=3, batch_norm=
        False, non_lin='tanh', dropout=0.0, first_block=False, last_block=
        False, skip_connection=False):
        super(Conv_Blocks, self).__init__()
        self.skip_connection = skip_connection
        self.last_block = last_block
        self.first_block = first_block
        self.Block = nn.Sequential()
        self.Block.add_module('Conv_1', nn.Conv2d(input_dim, output_dim,
            filter_size, 1, 1))
        if batch_norm:
            self.Block.add_module('BN_1', nn.BatchNorm2d(output_dim))
        if non_lin == 'tanh':
            self.Block.add_module('NonLin_1', nn.Tanh())
        elif non_lin == 'relu':
            self.Block.add_module('NonLin_1', nn.ReLU())
        elif non_lin == 'leakyrelu':
            self.Block.add_module('NonLin_1', nn.LeakyReLU())
        else:
            assert False, "non_lin = {} not valid: 'tanh', 'relu', 'leakyrelu'".format(
                non_lin)
        self.Block.add_module('Pool', nn.MaxPool2d(kernel_size=(2, 2),
            stride=(2, 2), dilation=(1, 1), ceil_mode=False))
        if dropout > 0:
            self.Block.add_module('Drop', nn.Dropout2d(dropout))

    def forward(self, x):
        if self.skip_connection:
            if not self.first_block:
                x, skip_con_list = x
            else:
                skip_con_list = []
        x = self.Block(x)
        if self.skip_connection:
            if not self.last_block:
                skip_con_list.append(x)
            x = [x, skip_con_list]
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
