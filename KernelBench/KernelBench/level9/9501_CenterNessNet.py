import math
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        norm = self.norm(x)
        relu = self.relu(norm)
        return relu


class DCNv2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels,
            *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[0
            ] * self.kernel_size[1] == offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[
            1] == mask.shape[1]
        return dcn_v2_conv(input, offset, mask, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.deformable_groups)


class DCN(DCNv2):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, dilation=1, deformable_groups=1):
        super(DCN, self).__init__(in_channels, out_channels, kernel_size,
            stride, padding, dilation, deformable_groups)
        channels_ = self.deformable_groups * 3 * self.kernel_size[0
            ] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels, channels_,
            kernel_size=self.kernel_size, stride=self.stride, padding=self.
            padding, bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return dcn_v2_conv(input, offset, mask, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.deformable_groups)


class CenterNessNet(nn.Module):

    def __init__(self, in_channels=256, feat_channels=256, stacked_convs=4,
        dcn_on_last_conv=False):
        super(CenterNessNet, self).__init__()
        self.stacked_convs = stacked_convs
        self.dcn_on_last_conv = dcn_on_last_conv
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self._init_layers()
        self.init_weight()

    def _init_layers(self):
        self._init_centerness_convs()
        self._init_centerness_predict()

    def normal_init(self, module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def init_weight(self):
        for m in self.centerness_convs.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif isinstance(m, nn.GroupNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    torch.nn.normal(m.weight.data, 0, 0.01)
                    m.bias.zero_()
        for m in self.centerness_predict.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _init_centerness_convs(self):
        self.centerness_convs = nn.Sequential()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = DCN(chn, self.feat_channels, kernel_size=(3, 3),
                    stride=1, padding=1, deformable_groups=1)
            else:
                conv_cfg = BasicBlock(chn, self.feat_channels, 3, 1, 1)
            self.centerness_convs.add_module(('centerness_' + str({0})).
                format(i), conv_cfg)

    def _init_centerness_predict(self):
        self.centerness_predict = nn.Sequential()
        predict = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.centerness_predict.add_module('centerness_predict', predict)

    def forward(self, x):
        convs = self.centerness_convs(x)
        predict = self.centerness_predict(convs)
        return predict


def get_inputs():
    return [torch.rand([4, 256, 64, 64])]


def get_init_inputs():
    return [[], {}]
