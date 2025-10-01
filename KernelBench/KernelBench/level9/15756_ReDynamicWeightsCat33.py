import math
import torch
import torch.utils.data
from torch import nn
from torch.nn.modules.utils import _pair


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, deformable_groups=1, bias=False):
        assert not bias
        super(DeformConv, self).__init__()
        self.with_bias = bias
        assert in_channels % groups == 0, 'in_channels {} cannot be divisible by groups {}'.format(
            in_channels, groups)
        assert out_channels % groups == 0, 'out_channels {} cannot be divisible by groups {}'.format(
            out_channels, groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels //
            self.groups, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, offset):
        return deform_conv(input, offset, self.weight, self.stride, self.
            padding, self.dilation, self.groups, self.deformable_groups)

    def __repr__(self):
        return ''.join(['{}('.format(self.__class__.__name__),
            'in_channels={}, '.format(self.in_channels),
            'out_channels={}, '.format(self.out_channels),
            'kernel_size={}, '.format(self.kernel_size), 'stride={}, '.
            format(self.stride), 'dilation={}, '.format(self.dilation),
            'padding={}, '.format(self.padding), 'groups={}, '.format(self.
            groups), 'deformable_groups={}, '.format(self.deformable_groups
            ), 'bias={})'.format(self.with_bias)])


class DeformUnfold(nn.Module):

    def __init__(self, kernel_size, stride=1, padding=0, dilation=1,
        deformable_groups=1, bias=False):
        assert not bias
        super(DeformUnfold, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

    def forward(self, input, offset):
        return deform_unfold(input, offset, self.kernel_size, self.stride,
            self.padding, self.dilation, self.deformable_groups)

    def __repr__(self):
        return ''.join(['{}('.format(self.__class__.__name__),
            'kernel_size={}, '.format(self.kernel_size), 'stride={}, '.
            format(self.stride), 'dilation={}, '.format(self.dilation),
            'padding={}, '.format(self.padding), 'deformable_groups={}, '.
            format(self.deformable_groups)])


class ReDynamicWeightsCat33(nn.Module):
    """' a rigrous implementation but slow

    """

    def __init__(self, channels, group=1, kernel=3, dilation=(1, 4, 8, 12),
        shuffle=False, deform=None):
        super(ReDynamicWeightsCat33, self).__init__()
        in_channel = channels
        if deform == 'deformatt':
            self.off_conva = nn.Conv2d(in_channel, 18, 3, padding=dilation[
                0], dilation=dilation[0], bias=False)
            self.off_convb = nn.Conv2d(in_channel, 18, 3, padding=dilation[
                1], dilation=dilation[1], bias=False)
            self.off_convc = nn.Conv2d(in_channel, 18, 3, padding=dilation[
                2], dilation=dilation[2], bias=False)
            self.off_convd = nn.Conv2d(in_channel, 18, 3, padding=dilation[
                3], dilation=dilation[3], bias=False)
            self.kernel_conva = DeformConv(in_channel, group * kernel *
                kernel + 9, kernel_size=3, padding=dilation[0], dilation=
                dilation[0], bias=False)
            self.kernel_convb = DeformConv(in_channel, group * kernel *
                kernel + 9, kernel_size=3, padding=dilation[1], dilation=
                dilation[1], bias=False)
            self.kernel_convc = DeformConv(in_channel, group * kernel *
                kernel + 9, kernel_size=3, padding=dilation[2], dilation=
                dilation[2], bias=False)
            self.kernel_convd = DeformConv(in_channel, group * kernel *
                kernel + 9, kernel_size=3, padding=dilation[3], dilation=
                dilation[3], bias=False)
            self.unfold1 = DeformUnfold(kernel_size=(3, 3), padding=
                dilation[0], dilation=dilation[0])
            self.unfold2 = DeformUnfold(kernel_size=(3, 3), padding=
                dilation[1], dilation=dilation[1])
            self.unfold3 = DeformUnfold(kernel_size=(3, 3), padding=
                dilation[2], dilation=dilation[2])
            self.unfold4 = DeformUnfold(kernel_size=(3, 3), padding=
                dilation[3], dilation=dilation[3])
        elif deform == 'deform':
            self.off_conva = nn.Conv2d(in_channel, 18, 3, padding=dilation[
                0], dilation=dilation[0], bias=False)
            self.off_convb = nn.Conv2d(in_channel, 18, 3, padding=dilation[
                1], dilation=dilation[1], bias=False)
            self.off_convc = nn.Conv2d(in_channel, 18, 3, padding=dilation[
                2], dilation=dilation[2], bias=False)
            self.off_convd = nn.Conv2d(in_channel, 18, 3, padding=dilation[
                3], dilation=dilation[3], bias=False)
            self.kernel_conva = DeformConv(in_channel, group * kernel *
                kernel, kernel_size=3, padding=dilation[0], dilation=
                dilation[0], bias=False)
            self.kernel_convb = DeformConv(in_channel, group * kernel *
                kernel, kernel_size=3, padding=dilation[1], dilation=
                dilation[1], bias=False)
            self.kernel_convc = DeformConv(in_channel, group * kernel *
                kernel, kernel_size=3, padding=dilation[2], dilation=
                dilation[2], bias=False)
            self.kernel_convd = DeformConv(in_channel, group * kernel *
                kernel, kernel_size=3, padding=dilation[3], dilation=
                dilation[3], bias=False)
            self.unfold1 = DeformUnfold(kernel_size=(3, 3), padding=
                dilation[0], dilation=dilation[0])
            self.unfold2 = DeformUnfold(kernel_size=(3, 3), padding=
                dilation[1], dilation=dilation[1])
            self.unfold3 = DeformUnfold(kernel_size=(3, 3), padding=
                dilation[2], dilation=dilation[2])
            self.unfold4 = DeformUnfold(kernel_size=(3, 3), padding=
                dilation[3], dilation=dilation[3])
        else:
            self.cata = nn.Conv2d(in_channel, group * kernel * kernel, 3,
                padding=dilation[0], dilation=dilation[0], bias=False)
            self.catb = nn.Conv2d(in_channel, group * kernel * kernel, 3,
                padding=dilation[1], dilation=dilation[1], bias=False)
            self.catc = nn.Conv2d(in_channel, group * kernel * kernel, 3,
                padding=dilation[2], dilation=dilation[2], bias=False)
            self.catd = nn.Conv2d(in_channel, group * kernel * kernel, 3,
                padding=dilation[3], dilation=dilation[3], bias=False)
            self.unfold1 = nn.Unfold(kernel_size=(3, 3), padding=dilation[0
                ], dilation=dilation[0])
            self.unfold2 = nn.Unfold(kernel_size=(3, 3), padding=dilation[1
                ], dilation=dilation[1])
            self.unfold3 = nn.Unfold(kernel_size=(3, 3), padding=dilation[2
                ], dilation=dilation[2])
            self.unfold4 = nn.Unfold(kernel_size=(3, 3), padding=dilation[3
                ], dilation=dilation[3])
        self.softmax = nn.Softmax(dim=-1)
        self.shuffle = shuffle
        self.deform = deform
        self.group = group
        self.K = kernel * kernel
        self.gamma1 = nn.Parameter(torch.FloatTensor(1).fill_(1.0))
        self.gamma2 = nn.Parameter(torch.FloatTensor(1).fill_(1.0))
        self.gamma3 = nn.Parameter(torch.FloatTensor(1).fill_(1.0))
        self.gamma4 = nn.Parameter(torch.FloatTensor(1).fill_(1.0))

    def forward(self, x):
        blur_depth = x
        N, C, H, W = x.size()
        R = C // self.group
        if self.deform == 'deformatt':
            offset1 = self.off_conva(blur_depth)
            offset2 = self.off_convb(blur_depth)
            offset3 = self.off_convc(blur_depth)
            offset4 = self.off_convd(blur_depth)
            xd_unfold1 = self.unfold1(blur_depth, offset1)
            xd_unfold2 = self.unfold2(blur_depth, offset2)
            xd_unfold3 = self.unfold3(blur_depth, offset3)
            xd_unfold4 = self.unfold4(blur_depth, offset4)
            dynamic_filter_att1 = self.kernel_conva(blur_depth, offset1)
            dynamic_filter_att2 = self.kernel_convb(blur_depth, offset2)
            dynamic_filter_att3 = self.kernel_convc(blur_depth, offset3)
            dynamic_filter_att4 = self.kernel_convd(blur_depth, offset4)
            dynamic_filter1 = dynamic_filter_att1[:, :9 * self.group, :, :]
            att1 = dynamic_filter_att1[:, -9:, :, :]
            att1 = att1.view(N, -1, H * W).view(N, 1, -1, H * W).permute(0,
                1, 3, 2).contiguous()
            att1 = self.softmax(att1)
            dynamic_filter2 = dynamic_filter_att2[:, :9 * self.group, :, :]
            att2 = dynamic_filter_att2[:, -9:, :, :]
            att2 = att2.view(N, -1, H * W).view(N, 1, -1, H * W).permute(0,
                1, 3, 2).contiguous()
            att2 = self.softmax(att2)
            dynamic_filter3 = dynamic_filter_att3[:, :9 * self.group, :, :]
            att3 = dynamic_filter_att3[:, -9:, :, :]
            att3 = att3.view(N, -1, H * W).view(N, 1, -1, H * W).permute(0,
                1, 3, 2).contiguous()
            att3 = self.softmax(att3)
            dynamic_filter4 = dynamic_filter_att4[:, :9 * self.group, :, :]
            att4 = dynamic_filter4[:, -9:, :, :]
            att4 = att4.view(N, -1, H * W).view(N, 1, -1, H * W).permute(0,
                1, 3, 2).contiguous()
            att4 = self.softmax(att4)
        elif self.deform == 'deform':
            offset1 = self.off_conva(blur_depth)
            offset2 = self.off_convb(blur_depth)
            offset3 = self.off_convc(blur_depth)
            offset4 = self.off_convd(blur_depth)
            xd_unfold1 = self.unfold1(blur_depth, offset1)
            xd_unfold2 = self.unfold2(blur_depth, offset2)
            xd_unfold3 = self.unfold3(blur_depth, offset3)
            xd_unfold4 = self.unfold4(blur_depth, offset4)
            dynamic_filter1 = self.kernel_conva(blur_depth, offset1)
            dynamic_filter2 = self.kernel_convb(blur_depth, offset2)
            dynamic_filter3 = self.kernel_convc(blur_depth, offset3)
            dynamic_filter4 = self.kernel_convd(blur_depth, offset4)
        else:
            dynamic_filter1 = self.cata(blur_depth)
            dynamic_filter2 = self.catb(blur_depth)
            dynamic_filter3 = self.catc(blur_depth)
            dynamic_filter4 = self.catd(blur_depth)
            xd_unfold1 = self.unfold1(blur_depth)
            xd_unfold2 = self.unfold2(blur_depth)
            xd_unfold3 = self.unfold3(blur_depth)
            xd_unfold4 = self.unfold4(blur_depth)
        if self.deform == 'deformatt':
            dynamic_filter1 = dynamic_filter1.view(N, self.group, self.K, -1
                ).permute(0, 1, 3, 2).contiguous()
            dynamic_filter2 = dynamic_filter2.view(N, self.group, self.K, -1
                ).permute(0, 1, 3, 2).contiguous()
            dynamic_filter3 = dynamic_filter3.view(N, self.group, self.K, -1
                ).permute(0, 1, 3, 2).contiguous()
            dynamic_filter4 = dynamic_filter4.view(N, self.group, self.K, -1
                ).permute(0, 1, 3, 2).contiguous()
        else:
            dynamic_filter1 = self.softmax(dynamic_filter1.view(N, self.
                group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1,
                self.K))
            dynamic_filter2 = self.softmax(dynamic_filter2.view(N, self.
                group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1,
                self.K))
            dynamic_filter3 = self.softmax(dynamic_filter3.view(N, self.
                group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1,
                self.K))
            dynamic_filter4 = self.softmax(dynamic_filter4.view(N, self.
                group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1,
                self.K))
        if self.training and self.shuffle:
            dynamic_filter1 = dynamic_filter1.view(N, self.group, H * W, self.K
                ).permute(1, 0, 2, 3).contiguous()
            idx1 = torch.randperm(self.group)
            dynamic_filter1 = dynamic_filter1[idx1].permute(1, 0, 2, 3
                ).contiguous().view(-1, self.K)
            dynamic_filter2 = dynamic_filter2.view(N, self.group, H * W, self.K
                ).permute(1, 0, 2, 3).contiguous()
            idx2 = torch.randperm(self.group)
            dynamic_filter2 = dynamic_filter2[idx2].permute(1, 0, 2, 3
                ).contiguous().view(-1, self.K)
            dynamic_filter3 = dynamic_filter3.view(N, self.group, H * W, self.K
                ).permute(1, 0, 2, 3).contiguous()
            idx3 = torch.randperm(self.group)
            dynamic_filter3 = dynamic_filter3[idx3].permute(1, 0, 2, 3
                ).contiguous().view(-1, self.K)
            dynamic_filter4 = dynamic_filter4.view(N, self.group, H * W, self.K
                ).permute(1, 0, 2, 3).contiguous()
            idx4 = torch.randperm(self.group)
            dynamic_filter4 = dynamic_filter4[idx4].permute(1, 0, 2, 3
                ).contiguous().view(-1, self.K)
        if self.deform == 'deformatt':
            dynamic_filter1 = dynamic_filter1 * att1
            dynamic_filter2 = dynamic_filter2 * att2
            dynamic_filter3 = dynamic_filter3 * att3
            dynamic_filter4 = dynamic_filter4 * att4
            dynamic_filter1 = dynamic_filter1.view(-1, self.K)
            dynamic_filter2 = dynamic_filter2.view(-1, self.K)
            dynamic_filter3 = dynamic_filter3.view(-1, self.K)
            dynamic_filter4 = dynamic_filter4.view(-1, self.K)
        xd_unfold1 = xd_unfold1.view(N, C, self.K, H * W).permute(0, 1, 3, 2
            ).contiguous().view(N, self.group, R, H * W, self.K).permute(0,
            1, 3, 2, 4).contiguous().view(N * self.group * H * W, R, self.K)
        xd_unfold2 = xd_unfold2.view(N, C, self.K, H * W).permute(0, 1, 3, 2
            ).contiguous().view(N, self.group, R, H * W, self.K).permute(0,
            1, 3, 2, 4).contiguous().view(N * self.group * H * W, R, self.K)
        xd_unfold3 = xd_unfold3.view(N, C, self.K, H * W).permute(0, 1, 3, 2
            ).contiguous().view(N, self.group, R, H * W, self.K).permute(0,
            1, 3, 2, 4).contiguous().view(N * self.group * H * W, R, self.K)
        xd_unfold4 = xd_unfold4.view(N, C, self.K, H * W).permute(0, 1, 3, 2
            ).contiguous().view(N, self.group, R, H * W, self.K).permute(0,
            1, 3, 2, 4).contiguous().view(N * self.group * H * W, R, self.K)
        out1 = torch.bmm(xd_unfold1, dynamic_filter1.unsqueeze(2))
        out1 = out1.view(N, self.group, H * W, R).permute(0, 1, 3, 2
            ).contiguous().view(N, self.group * R, H * W).view(N, self.
            group * R, H, W)
        out2 = torch.bmm(xd_unfold2, dynamic_filter2.unsqueeze(2))
        out2 = out2.view(N, self.group, H * W, R).permute(0, 1, 3, 2
            ).contiguous().view(N, self.group * R, H * W).view(N, self.
            group * R, H, W)
        out3 = torch.bmm(xd_unfold3, dynamic_filter3.unsqueeze(2))
        out3 = out3.view(N, self.group, H * W, R).permute(0, 1, 3, 2
            ).contiguous().view(N, self.group * R, H * W).view(N, self.
            group * R, H, W)
        out4 = torch.bmm(xd_unfold4, dynamic_filter4.unsqueeze(2))
        out4 = out4.view(N, self.group, H * W, R).permute(0, 1, 3, 2
            ).contiguous().view(N, self.group * R, H * W).view(N, self.
            group * R, H, W)
        out = (x + self.gamma1 * out1 + self.gamma2 * out2 + self.gamma3 *
            out3 + self.gamma4 * out4)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
