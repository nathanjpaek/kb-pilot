import torch
import torch.nn as nn


class encoder_block(nn.Module):

    def __init__(self, input_feature, output_feature, use_dropout):
        super(encoder_block, self).__init__()
        self.conv_input = nn.Conv3d(input_feature, output_feature, 3, 1, 1, 1)
        self.conv_inblock1 = nn.Conv3d(output_feature, output_feature, 3, 1,
            1, 1)
        self.conv_inblock2 = nn.Conv3d(output_feature, output_feature, 3, 1,
            1, 1)
        self.conv_pooling = nn.Conv3d(output_feature, output_feature, 2, 2,
            1, 1)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.2)

    def apply_dropout(self, input):
        if self.use_dropout:
            return self.dropout(input)
        else:
            return input

    def forward(self, x):
        output = self.conv_input(x)
        output = self.apply_dropout(self.prelu1(output))
        output = self.apply_dropout(self.prelu2(self.conv_inblock1(output)))
        output = self.apply_dropout(self.prelu3(self.conv_inblock2(output)))
        return self.prelu4(self.conv_pooling(output))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_feature': 4, 'output_feature': 4, 'use_dropout': 0.5}]
