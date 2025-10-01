import torch
import torch.nn as nn


class ReconstructionLayer(nn.Module):

    def __init__(self, ratio, input_channel, output_channel):
        super(ReconstructionLayer, self).__init__()
        self.deconv_features = nn.ConvTranspose1d(input_channel,
            output_channel, ratio, stride=ratio)

    def forward(self, x):
        feature = self.deconv_features(x.permute(0, 2, 1)).permute(0, 2, 1)
        return feature


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'ratio': 4, 'input_channel': 4, 'output_channel': 4}]
