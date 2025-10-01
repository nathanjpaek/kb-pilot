import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple


def conv1x1(in_planes, out_planes, stride=1, args=None, force_fp=False):
    """1x1 convolution"""
    if args is not None and hasattr(args, 'keyword'):
        return custom_conv(in_planes, out_planes, kernel_size=1, stride=
            stride, bias=False, args=args, force_fp=force_fp)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=
            stride, bias=False)


class quantization(nn.Module):

    def __init__(self, args=None, tag='fm', shape=[], feature_stride=None,
        logger=None):
        super(quantization, self).__init__()
        self.index = -1
        self.tag = tag
        self.logger = logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
        if args is None:
            self.enable = False
            return
        self.args = args
        self.shape = shape
        self.feature_stride = feature_stride
        self.enable = getattr(args, tag + '_enable', False)
        self.adaptive = getattr(self.args, self.tag + '_adaptive', 'none')
        self.grad_scale = getattr(self.args, self.tag + '_grad_scale', 'none')
        self.grad_type = getattr(args, tag + '_grad_type', 'none')
        self.custom = getattr(args, tag + '_custom', 'none')
        self.bit = getattr(args, tag + '_bit', None)
        self.num_levels = getattr(args, tag + '_level', None)
        self.half_range = getattr(args, tag + '_half_range', None)
        self.scale = getattr(args, tag + '_scale', 0.5)
        self.ratio = getattr(args, tag + '_ratio', 1)
        self.correlate = getattr(args, tag + '_correlate', -1)
        self.quant_group = getattr(args, tag + '_quant_group', None)
        self.boundary = getattr(self.args, self.tag + '_boundary', None)
        if self.bit is None:
            self.bit = 32
        if self.num_levels is None or self.num_levels <= 0:
            self.num_levels = int(2 ** self.bit)
        self.bit = int(self.bit)
        if self.half_range is None:
            self.half_range = tag == 'fm'
        else:
            self.half_range = bool(self.half_range)
        if self.quant_group == 0:
            self.quant_group = None
        if self.quant_group is not None:
            if self.quant_group < 0:
                if shape[0] * shape[1] % -self.quant_group != 0:
                    self.quant_group = None
                else:
                    self.quant_group = shape[0] * shape[1] / -self.quant_group
            elif shape[0] * shape[1] % self.quant_group != 0:
                self.quant_group = None
        if self.quant_group is not None:
            self.quant_group = int(self.quant_group)
        else:
            self.quant_group = shape[0] if self.tag == 'wt' else 1
        self.fan = 1
        for i in range(len(self.shape) - 1):
            self.fan *= self.shape[i + 1]
        if 'proxquant' in self.args.keyword:
            self.prox = 0
        if not self.enable:
            return
        self.logger.info(
            'half_range({}), bit({}), num_levels({}), quant_group({}) boundary({}) scale({}) ratio({}) fan({}) tag({})'
            .format(self.half_range, self.bit, self.num_levels, self.
            quant_group, self.boundary, self.scale, self.ratio, self.fan,
            self.tag))
        self.method = 'none'
        self.times = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.learning_rate = 1
        self.init_learning_rate = 1
        self.progressive = False
        self.init()

    def init(self):
        if ('lq' in self.args.keyword or 'alq' in self.args.keyword or 
            'popcount' in self.args.keyword):
            if not hasattr(self, 'num_levels'):
                self.num_levels = 2 ** self.bit
            if self.num_levels > 256:
                raise RuntimeError(
                    'currently not support more than 8 bit quantization')
            if self.num_levels == 3:
                self.bit = 1
                self.logger.info('update %s_bit %r' % (self.tag, self.bit))
            self.method = 'lqnet'
            if 'lq' in self.args.keyword:
                self.lq_net_init()
                self.quant_fm = alqnet.LqNet_fm
                self.quant_wt = alqnet.LqNet_wt
            init_thrs_multiplier = []
            for i in range(1, self.num_levels):
                thrs_multiplier_i = [(0.0) for j in range(self.num_levels)]
                if not self.half_range:
                    if i < self.num_levels / 2:
                        thrs_multiplier_i[i - 1] = 1 - self.scale
                        thrs_multiplier_i[i] = self.scale
                    elif i > self.num_levels / 2:
                        thrs_multiplier_i[i - 1] = self.scale
                        thrs_multiplier_i[i] = 1 - self.scale
                    else:
                        thrs_multiplier_i[i - 1] = 0.5
                        thrs_multiplier_i[i] = 0.5
                else:
                    thrs_multiplier_i[i - 1] = self.scale
                    thrs_multiplier_i[i] = 1 - self.scale
                init_thrs_multiplier.append(thrs_multiplier_i)
            self.thrs_multiplier = nn.Parameter(torch.zeros(self.num_levels -
                1, self.num_levels), requires_grad=False)
            self.thrs_multiplier.data = torch.FloatTensor(init_thrs_multiplier)
            if 'debug' in self.args.keyword:
                self.logger.info('self.thrs_multiplier: {}'.format(self.
                    thrs_multiplier))
        if 'dorefa' in self.args.keyword or 'pact' in self.args.keyword:
            self.method = 'dorefa'
            if self.boundary is None:
                self.boundary = 1.0
                self.logger.info('update %s_boundary %r' % (self.tag, self.
                    boundary))
            if self.tag == 'fm':
                if 'pact' in self.args.keyword:
                    self.quant_fm = dorefa.qfn
                    self.clip_val = nn.Parameter(torch.Tensor([self.boundary]))
                elif 'lsq' in self.args.keyword or 'fm_lsq' in self.args.keyword:
                    self.clip_val = nn.Parameter(torch.Tensor([self.boundary]))
                    self.quant_fm = dorefa.LSQ
                else:
                    self.quant_fm = dorefa.qfn
                    self.clip_val = self.boundary
            elif 'lsq' in self.args.keyword or 'wt_lsq' in self.args.keyword:
                if self.shape[0] == 1:
                    raise RuntimeError(
                        'Quantization for linear layer not provided')
                else:
                    self.clip_val = nn.Parameter(torch.zeros(self.
                        quant_group, 1, 1, 1))
                self.clip_val.data.fill_(self.boundary)
                self.quant_wt = dorefa.LSQ
            elif 'wt_bin' in self.args.keyword and self.num_levels == 2:
                self.quant_wt = dorefa.DorefaParamsBinarizationSTE
            else:
                self.quant_wt = dorefa.qfn
                self.clip_val = self.boundary
        if 'xnor' in self.args.keyword:
            if self.tag == 'fm':
                self.quant_fm = xnor.XnorActivation
                if 'debug' in self.args.keyword:
                    self.logger.info('debug: tag: {} custom: {}, grad_type {}'
                        .format(self.tag, self.custom, self.grad_type))
            else:
                if 'debug' in self.args.keyword:
                    self.logger.info('debug: tag: {} custom: {}, grad_type {}'
                        .format(self.tag, self.custom, self.grad_type))
                self.quant_wt = xnor.XnorWeight
                if 'gamma' in self.args.keyword:
                    self.gamma = nn.Parameter(torch.ones(self.quant_group, 
                        1, 1, 1))

    def update_quantization(self, **parameters):
        index = self.index
        if 'index' in parameters:
            index = parameters['index']
        if index != self.index:
            self.index = index
            self.logger.info('update %s_index %r' % (self.tag, self.index))
        if not self.enable:
            return
        if self.method == 'dorefa':
            if 'progressive' in parameters:
                self.progressive = parameters['progressive']
                self.logger.info('update %s_progressive %r' % (self.tag,
                    self.progressive))
            if self.progressive:
                bit = self.bit
                num_level = self.num_levels
                if self.tag == 'fm':
                    if 'fm_bit' in parameters:
                        bit = parameters['fm_bit']
                    if 'fm_level' in parameters:
                        num_level = parameters['fm_level']
                else:
                    if 'wt_bit' in parameters:
                        bit = parameters['wt_bit']
                    if 'wt_level' in parameters:
                        num_level = parameters['wt_level']
                if bit != self.bit:
                    self.bit = bit
                    num_level = 2 ** self.bit
                    self.logger.info('update %s_bit %r' % (self.tag, self.bit))
                if num_level != self.num_levels:
                    self.num_levels = num_level
                    self.logger.info('update %s_level %r' % (self.tag, self
                        .num_levels))
        if self.method == 'lqnet':
            pass

    def init_based_on_warmup(self, data=None):
        if not self.enable:
            return
        with torch.no_grad():
            if self.method == 'dorefa' and data is not None:
                pass
        return

    def init_based_on_pretrain(self, weight=None):
        if not self.enable:
            return
        with torch.no_grad():
            if self.method == 'dorefa' and 'non-uniform' in self.args.keyword:
                pass
        return

    def update_bias(self, basis=None):
        if not self.training:
            return
        if 'custom-update' not in self.args.keyword:
            self.basis.data = basis
            self.times.data = self.times.data + 1
        else:
            self.basis.data = self.basis.data * self.times + self.auxil
            self.times.data = self.times.data + 1
            self.basis.data = self.basis.data / self.times

    def quantization_value(self, x, y):
        if self.times.data < self.args.stable:
            self.init_based_on_warmup(x)
            return x
        elif 'proxquant' in self.args.keyword:
            return x * self.prox + y * (1 - self.prox)
        else:
            if ('probe' in self.args.keyword and self.index >= 0 and not
                self.training and self.tag == 'fm'):
                for item in self.args.probe_list:
                    if 'before-quant' == item:
                        torch.save(x, 'log/{}-activation-latent.pt'.format(
                            self.index))
                    elif 'after-quant' == item:
                        torch.save(y, 'log/{}-activation-quant.pt'.format(
                            self.index))
                    elif hasattr(self, item):
                        torch.save(getattr(self, item),
                            'log/{}-activation-{}.pt'.format(self.index, item))
                self.index = -1
            return y

    def forward(self, x):
        if not self.enable:
            return x
        if self.method == 'lqnet':
            if self.tag == 'fm':
                y, basis = self.quant_fm.apply(x, self.basis, self.
                    codec_vector, self.codec_index, self.thrs_multiplier,
                    self.training, self.half_range, self.auxil, self.adaptive)
            else:
                y, basis = self.quant_wt.apply(x, self.basis, self.
                    codec_vector, self.codec_index, self.thrs_multiplier,
                    self.training, self.half_range, self.auxil, self.adaptive)
            self.update_bias(basis)
            return self.quantization_value(x, y)
        if 'xnor' in self.args.keyword:
            if self.tag == 'fm':
                y = self.quant_fm.apply(x, self.custom, self.grad_type)
            else:
                if self.adaptive == 'var-mean':
                    std, mean = torch.std_mean(x.data.reshape(self.
                        quant_group, -1, 1, 1, 1), 1)
                    x = (x - mean) / (std + __EPS__)
                y = self.quant_wt.apply(x, self.quant_group, self.grad_type)
                if 'gamma' in self.args.keyword:
                    y = y * self.gamma
            return self.quantization_value(x, y)
        if self.method == 'dorefa':
            if self.tag == 'fm':
                if 'lsq' in self.args.keyword or 'fm_lsq' in self.args.keyword:
                    if self.half_range:
                        y = x / self.clip_val
                        y = torch.clamp(y, min=0, max=1)
                        y = self.quant_fm.apply(y, self.num_levels - 1)
                        y = y * self.clip_val
                    else:
                        y = x / self.clip_val
                        y = torch.clamp(y, min=-1, max=1)
                        y = (y + 1.0) / 2.0
                        y = self.quant_fm.apply(y, self.num_levels - 1)
                        y = y * 2.0 - 1.0
                        y = y * self.clip_val
                elif 'pact' in self.args.keyword:
                    y = torch.clamp(x, min=0)
                    y = torch.where(y < self.clip_val, y, self.clip_val)
                    y = self.quant_fm.apply(y, self.num_levels, self.
                        clip_val.detach(), self.adaptive)
                else:
                    y = torch.clamp(x, min=0, max=self.clip_val)
                    y = self.quant_fm.apply(y, self.num_levels, self.
                        clip_val, self.adaptive)
            else:
                if self.adaptive == 'var-mean':
                    std, mean = torch.std_mean(x.data.reshape(self.
                        quant_group, -1, 1, 1, 1), 1)
                    x = (x - mean) / (std + __EPS__)
                if 'lsq' in self.args.keyword or 'wt_lsq' in self.args.keyword:
                    y = x / self.clip_val
                    y = torch.clamp(y, min=-1, max=1)
                    y = (y + 1.0) / 2.0
                    y = self.quant_wt.apply(y, self.num_levels - 1)
                    y = y * 2.0 - 1.0
                    y = y * self.clip_val
                elif 'wt_bin' in self.args.keyword and self.num_levels == 2:
                    y = self.quant_wt.apply(x, self.adaptive)
                else:
                    y = torch.tanh(x)
                    y = y / (2 * y.abs().max()) + 0.5
                    y = 2 * self.quant_wt.apply(y, self.num_levels, self.
                        clip_val, self.adaptive) - 1
            self.times.data = self.times.data + 1
            return self.quantization_value(x, y)
        raise RuntimeError('Should not reach here in quant.py')

    def lq_net_init(self):
        self.basis = nn.Parameter(torch.ones(self.bit, self.quant_group),
            requires_grad=False)
        self.auxil = nn.Parameter(torch.zeros(self.bit, self.quant_group),
            requires_grad=False)
        self.codec_vector = nn.Parameter(torch.ones(self.num_levels, self.
            bit), requires_grad=False)
        self.codec_index = nn.Parameter(torch.ones(self.num_levels, dtype=
            torch.int), requires_grad=False)
        init_basis = []
        NORM_PPF_0_75 = 0.6745
        if self.tag == 'fm':
            base = NORM_PPF_0_75 * 2.0 / 2 ** (self.bit - 1)
        else:
            base = NORM_PPF_0_75 * (2.0 / self.fan) ** 0.5 / 2 ** (self.bit - 1
                )
        for i in range(self.bit):
            init_basis.append([(2 ** i * base) for j in range(self.
                quant_group)])
        self.basis.data = torch.FloatTensor(init_basis)
        init_level_multiplier = []
        for i in range(self.num_levels):
            level_multiplier_i = [(0.0) for j in range(self.bit)]
            level_number = i
            for j in range(self.bit):
                binary_code = level_number % 2
                if binary_code == 0 and not self.half_range:
                    binary_code = -1
                level_multiplier_i[j] = float(binary_code)
                level_number = level_number // 2
            init_level_multiplier.append(level_multiplier_i)
        self.codec_vector.data = torch.FloatTensor(init_level_multiplier)
        init_codec_index = []
        for i in range(self.num_levels):
            init_codec_index.append(i)
        self.codec_index.data = torch.IntTensor(init_codec_index)


class custom_conv(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=False, args=None, force_fp=
        False, feature_stride=1):
        super(custom_conv, self).__init__(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.args = args
        self.force_fp = force_fp
        if not self.force_fp:
            self.pads = padding
            self.padding = 0, 0
            self.quant_activation = quantization(args, 'fm', [1,
                in_channels, 1, 1], feature_stride=feature_stride)
            self.quant_weight = quantization(args, 'wt', [out_channels,
                in_channels, kernel_size, kernel_size])
            self.padding_after_quant = getattr(args, 'padding_after_quant',
                False) if args is not None else False

    def init_after_load_pretrain(self):
        if not self.force_fp:
            self.quant_weight.init_based_on_pretrain(self.weight.data)
            self.quant_activation.init_based_on_pretrain()

    def update_quantization_parameter(self, **parameters):
        if not self.force_fp:
            self.quant_activation.update_quantization(**parameters)
            self.quant_weight.update_quantization(**parameters)

    def forward(self, inputs):
        if not self.force_fp:
            weight = self.quant_weight(self.weight)
            if self.padding_after_quant:
                inputs = self.quant_activation(inputs)
                inputs = F.pad(inputs, _quadruple(self.pads), 'constant', 0)
            else:
                inputs = F.pad(inputs, _quadruple(self.pads), 'constant', 0)
                inputs = self.quant_activation(inputs)
        else:
            weight = self.weight
        output = F.conv2d(inputs, weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)
        return output


class Prone(nn.Module):

    def __init__(self, out_channel, in_channel=3, stride=4, group=1,
        kernel_size=3, force_fp=True, args=None, feature_stride=1, keepdim=True
        ):
        super(Prone, self).__init__()
        self.stride = stride * 2
        self.in_channel = in_channel * self.stride * self.stride
        self.out_channel = out_channel * 4
        self.keepdim = keepdim
        self.conv = conv1x1(self.in_channel, self.out_channel, args=args,
            force_fp=force_fp)

    def forward(self, x):
        B, C, H, W = x.shape
        if H % self.stride != 0:
            pad = (self.stride - H % self.stride) // 2
            x = F.pad(x, _quadruple(pad), mode='constant', value=0)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.stride, self.stride, W // self.stride,
            self.stride)
        x = x.transpose(4, 3).reshape(B, C, 1, H // self.stride, W // self.
            stride, self.stride * self.stride)
        x = x.transpose(2, 5).reshape(B, C * self.stride * self.stride, H //
            self.stride, W // self.stride)
        x = self.conv(x)
        if self.keepdim:
            B, C, H, W = x.shape
            x = x.reshape(B, C // 4, 4, H, W, 1)
            x = x.transpose(2, 5).reshape(B, C // 4, H, W, 2, 2)
            x = x.transpose(4, 3).reshape(B, C // 4, H * 2, W * 2)
        return x


def get_inputs():
    return [torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {'out_channel': 4}]
