import torch
import torch.nn as nn


class UpConv_Blocks(nn.Module):

    def __init__(self, input_dim, output_dim, filter=4, padding=1,
        first_block=False, last_block=False, batch_norm=False, non_lin=
        'relu', dropout=0, skip_connection=False):
        super(UpConv_Blocks, self).__init__()
        self.Block = nn.Sequential()
        self.skip_connection = skip_connection
        self.first_block = first_block
        self.last_block = last_block
        if self.skip_connection and not self.first_block:
            input_dim *= 2
        else:
            pass
        self.Block.add_module('UpConv', nn.ConvTranspose2d(input_dim,
            output_dim, filter, 2, padding))
        if not last_block:
            if batch_norm:
                self.Block.add_module('BN_up', nn.BatchNorm2d(output_dim))
            if non_lin == 'tanh':
                self.Block.add_module('NonLin_up', nn.Tanh())
            elif non_lin == 'relu':
                self.Block.add_module('NonLin_up', nn.ReLU())
            elif non_lin == 'leakyrelu':
                self.Block.add_module('NonLin_up', nn.LeakyReLU())
            if dropout > 0:
                self.Block.add_module('Drop_up', nn.Dropout2d(dropout))

    def forward(self, x):
        if self.skip_connection:
            x, skip_con_list = x
            if not self.first_block:
                x = torch.cat((x, skip_con_list.pop(-1)), -3)
        x = self.Block(x)
        if self.skip_connection and not self.last_block:
            x = [x, skip_con_list]
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
