import torch
import numpy as np
import torch.nn as nn


class SqueezeLayer(nn.Module):

    def __init__(self, factor):
        super(SqueezeLayer, self).__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False, **kwargs):
        if not reverse:
            assert input.size(-1) % self.factor == 0
            output = input.view(input.size(0), input.size(1), -1, self.factor)
            output = output.permute(0, 1, 3, 2).contiguous()
            output = output.view(input.size(0), -1, input.size(-1) // self.
                factor)
            return output, logdet
        else:
            assert input.size(1) % self.factor == 0
            output = input.view(input.size(0), -1, self.factor, input.size(-1))
            output = output.permute(0, 1, 3, 2).contiguous()
            output = output.view(input.size(0), input.size(1) // self.
                factor, -1)
            return output, logdet


class UpsampleNet(nn.Module):

    def __init__(self, upsample_factor, upsample_method='duplicate',
        squeeze_factor=8):
        super(UpsampleNet, self).__init__()
        self.upsample_factor = upsample_factor
        self.upsample_method = upsample_method
        self.squeeze_factor = squeeze_factor
        if upsample_method == 'duplicate':
            upsample_factor = int(np.prod(upsample_factor))
            self.upsample = nn.Upsample(scale_factor=upsample_factor, mode=
                'nearest')
        elif upsample_method == 'transposed_conv2d':
            if not isinstance(upsample_factor, list):
                raise ValueError(
                    'You must specify upsample_factor as a list when used with transposed_conv2d'
                    )
            freq_axis_kernel_size = 3
            self.upsample_conv = nn.ModuleList()
            for s in upsample_factor:
                freq_axis_padding = (freq_axis_kernel_size - 1) // 2
                conv = nn.ConvTranspose2d(1, 1, (freq_axis_kernel_size, 2 *
                    s), padding=(freq_axis_padding, s // 2), dilation=1,
                    stride=(1, s))
                self.upsample_conv.append(conv)
                self.upsample_conv.append(nn.LeakyReLU(negative_slope=0.4,
                    inplace=True))
        else:
            raise ValueError('{} upsampling is not supported'.format(self.
                _upsample_method))
        self.squeeze_layer = SqueezeLayer(squeeze_factor)

    def forward(self, input):
        if self.upsample_method == 'duplicate':
            output = self.upsample(input)
        elif self.upsample_method == 'transposed_conv2d':
            output = input.unsqueeze(1)
            for layer in self.upsample_conv:
                output = layer(output)
            output = output.squeeze(1)
        output = self.squeeze_layer(output)[0]
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'upsample_factor': 4}]
