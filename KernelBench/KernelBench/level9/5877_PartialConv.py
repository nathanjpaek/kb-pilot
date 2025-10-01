import torch
import torch.nn as nn
import torch.onnx


class PartialConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv, self).__init__()
        self.feature_conv = nn.Conv2d(in_channels, out_channels,
            kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding,
            dilation, groups, bias=False)
        self.window_size = self.mask_conv.kernel_size[0
            ] * self.mask_conv.kernel_size[1]
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = self.feature_conv(x)
        if self.feature_conv.bias is not None:
            output_bias = self.feature_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output, device=x.device)
        with torch.no_grad():
            ones = torch.ones(1, 1, x.size(2), x.size(3), device=x.device)
            output_mask = self.mask_conv(ones)
            output_mask = self.window_size / output_mask
        output = (output - output_bias) * output_mask + output_bias
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
