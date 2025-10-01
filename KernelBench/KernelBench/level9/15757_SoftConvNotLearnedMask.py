import math
import torch
import torch.nn as nn


def weights_init(init_type='gaussian'):

    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0
            ) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, 'Unsupported initialization: {}'.format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_fun


class SoftConvNotLearnedMask(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, 1, bias)
        self.mask_update_conv = nn.Conv2d(in_channels, out_channels,
            kernel_size, stride, padding, dilation, 1, False)
        self.input_conv.apply(weights_init('xavier'))

    def forward(self, input, mask):
        output = self.input_conv(input * mask)
        with torch.no_grad():
            self.mask_update_conv.weight = torch.nn.Parameter(self.
                input_conv.weight.abs())
            filters, _, _, _ = self.mask_update_conv.weight.shape
            k = self.mask_update_conv.weight.view((filters, -1)).sum(1)
            norm = k.view(1, -1, 1, 1).repeat(mask.shape[0], 1, 1, 1)
            new_mask = self.mask_update_conv(mask) / (norm + 1e-06)
        return output, new_mask


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
