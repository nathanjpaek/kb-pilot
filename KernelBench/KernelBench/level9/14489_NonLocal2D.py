import math
import torch
import torch.utils.data
from torch import nn
from torch.nn.modules.utils import _pair


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, 'GroupNorm: can only specify G or C/G.'
    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, 'dim: {}, dim_per_gp: {}'.format(dim,
            dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, 'dim: {}, num_groups: {}'.format(dim,
            num_groups)
        group_gn = num_groups
    return group_gn


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = cfg.MODEL.GROUP_NORM.DIM_PER_GP // divisor
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS // divisor
    eps = cfg.MODEL.GROUP_NORM.EPSILON
    return torch.nn.GroupNorm(get_group_gn(out_channels, dim_per_gp,
        num_groups), out_channels, eps, affine)


def conv_with_kaiming_uniform(use_gn=False, use_relu=False, use_deformable=
    False, use_bn=False):

    def make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1
        ):
        if use_deformable:
            conv_func = DFConv2d
        else:
            conv_func = Conv2d
        conv = conv_func(in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation, bias=not (use_gn or use_bn))
        if not use_deformable:
            nn.init.kaiming_uniform_(conv.weight, a=1)
            if not (use_gn or use_bn):
                nn.init.constant_(conv.bias, 0)
        module = [conv]
        if use_gn:
            module.append(group_norm(out_channels))
        elif use_bn:
            module.append(nn.BatchNorm2d(out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv
    return make_conv


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, deformable_groups=1, bias=False):
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
        if self.with_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.with_bias:
            torch.nn.init.constant_(self.bias, 0.0)

    def forward(self, input, offset):
        y = deform_conv(input, offset, self.weight, self.stride, self.
            padding, self.dilation, self.groups, self.deformable_groups)
        if self.with_bias:
            assert len(y.size()) == 4
            y = y + self.bias.reshape(1, -1, 1, 1)
        return y

    def __repr__(self):
        return ''.join(['{}('.format(self.__class__.__name__),
            'in_channels={}, '.format(self.in_channels),
            'out_channels={}, '.format(self.out_channels),
            'kernel_size={}, '.format(self.kernel_size), 'stride={}, '.
            format(self.stride), 'dilation={}, '.format(self.dilation),
            'padding={}, '.format(self.padding), 'groups={}, '.format(self.
            groups), 'deformable_groups={}, '.format(self.deformable_groups
            ), 'bias={})'.format(self.with_bias)])


class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = _pair(padding)
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels //
            groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input, offset, mask):
        return modulated_deform_conv(input, offset, mask, self.weight, self
            .bias, self.stride, self.padding, self.dilation, self.groups,
            self.deformable_groups)

    def __repr__(self):
        return ''.join(['{}('.format(self.__class__.__name__),
            'in_channels={}, '.format(self.in_channels),
            'out_channels={}, '.format(self.out_channels),
            'kernel_size={}, '.format(self.kernel_size), 'stride={}, '.
            format(self.stride), 'dilation={}, '.format(self.dilation),
            'padding={}, '.format(self.padding), 'groups={}, '.format(self.
            groups), 'deformable_groups={}, '.format(self.deformable_groups
            ), 'bias={})'.format(self.with_bias)])


class _NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(torch.nn.Conv2d):

    def forward(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).forward(x)
        output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // d + 1) for i,
            p, di, k, d in zip(x.shape[-2:], self.padding, self.dilation,
            self.kernel_size, self.stride)]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class DFConv2d(nn.Module):
    """Deformable convolutional layer"""

    def __init__(self, in_channels, out_channels, with_modulated_dcn=True,
        kernel_size=3, stride=1, groups=1, dilation=1, deformable_groups=1,
        bias=False, padding=None):
        super(DFConv2d, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            assert isinstance(stride, (list, tuple))
            assert isinstance(dilation, (list, tuple))
            assert len(kernel_size) == 2
            assert len(stride) == 2
            assert len(dilation) == 2
            padding = dilation[0] * (kernel_size[0] - 1) // 2, dilation[1] * (
                kernel_size[1] - 1) // 2
            offset_base_channels = kernel_size[0] * kernel_size[1]
        else:
            padding = dilation * (kernel_size - 1) // 2
            offset_base_channels = kernel_size * kernel_size
        if with_modulated_dcn:
            offset_channels = offset_base_channels * 3
            conv_block = ModulatedDeformConv
        else:
            offset_channels = offset_base_channels * 2
            conv_block = DeformConv
        self.offset = Conv2d(in_channels, deformable_groups *
            offset_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=1, dilation=dilation)
        for l in [self.offset]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            torch.nn.init.constant_(l.bias, 0.0)
        self.conv = conv_block(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, deformable_groups=deformable_groups, bias=bias)
        self.with_modulated_dcn = with_modulated_dcn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.offset_split = offset_base_channels * deformable_groups * 2

    def forward(self, x, return_offset=False):
        if x.numel() > 0:
            if not self.with_modulated_dcn:
                offset_mask = self.offset(x)
                x = self.conv(x, offset_mask)
            else:
                offset_mask = self.offset(x)
                offset = offset_mask[:, :self.offset_split, :, :]
                mask = offset_mask[:, self.offset_split:, :, :].sigmoid()
                x = self.conv(x, offset, mask)
            if return_offset:
                return x, offset_mask
            return x
        output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // d + 1) for i,
            p, di, k, d in zip(x.shape[-2:], self.padding, self.dilation,
            self.kernel_size, self.stride)]
        output_shape = [x.shape[0], self.conv.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class NonLocal2D(nn.Module):
    """Non-local module.
    See https://arxiv.org/abs/1711.07971 for details.
    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio.
        use_scale (bool): Whether to scale pairwise_weight by 1/inter_channels.
        mode (str): Options are `embedded_gaussian` and `dot_product`.
    """

    def __init__(self, in_channels, reduction=2, use_scale=True, use_gn=
        False, use_deformable=False, mode='embedded_gaussian'):
        super(NonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product']
        ConvModule = conv_with_kaiming_uniform()
        last_conv = conv_with_kaiming_uniform(use_gn=use_gn, use_deformable
            =use_deformable)
        self.g = ConvModule(self.in_channels, self.inter_channels,
            kernel_size=1)
        self.theta = ConvModule(self.in_channels, self.inter_channels,
            kernel_size=1)
        self.phi = ConvModule(self.in_channels, self.inter_channels,
            kernel_size=1)
        self.conv_out = last_conv(self.inter_channels, self.in_channels,
            kernel_size=1)

    def embedded_gaussian(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            pairwise_weight /= theta_x.shape[-1] ** -0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x):
        n, _, h, w = x.shape
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(n, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(n, self.inter_channels, -1)
        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(theta_x, phi_x)
        y = torch.matmul(pairwise_weight, g_x)
        y = y.permute(0, 2, 1).reshape(n, self.inter_channels, h, w)
        output = x + self.conv_out(y)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
