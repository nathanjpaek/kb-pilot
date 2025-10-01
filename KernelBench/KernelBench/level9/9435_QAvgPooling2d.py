from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F


def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    qmin = 0
    qmax = 2 ** num_bits - 1
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmax - max_val / scale
    if zero_point < qmin:
        zero_point = torch.tensor([qmin], dtype=torch.float32)
    elif zero_point > qmax:
        zero_point = torch.tensor([qmax], dtype=torch.float32)
    zero_point.round_()
    return scale, zero_point


def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False):
    if signed:
        qmin = -2.0 ** (num_bits - 1)
        qmax = 2.0 ** (num_bits - 1) - 1
    else:
        qmin = 0.0
        qmax = 2.0 ** num_bits - 1.0
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    return q_x


class FakeQuantize(Function):

    @staticmethod
    def forward(ctx, x, qparam):
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class QParam(nn.Module):

    def __init__(self, num_bits=8):
        super(QParam, self).__init__()
        self.num_bits = num_bits
        scale = torch.tensor([], requires_grad=False)
        zero_point = torch.tensor([], requires_grad=False)
        min = torch.tensor([], requires_grad=False)
        max = torch.tensor([], requires_grad=False)
        self.register_buffer('scale', scale)
        self.register_buffer('zero_point', zero_point)
        self.register_buffer('min', min)
        self.register_buffer('max', max)

    def update(self, tensor):
        if self.max.nelement() == 0 or self.max.data < tensor.max().data:
            self.max.data = tensor.max().data
        self.max.clamp_(min=0)
        if self.min.nelement() == 0 or self.min.data > tensor.min().data:
            self.min.data = tensor.min().data
        self.min.clamp_(max=0)
        self.scale, self.zero_point = calcScaleZeroPoint(self.min, self.max,
            self.num_bits)

    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.scale, self.zero_point,
            num_bits=self.num_bits)

    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.scale, self.zero_point)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        key_names = ['scale', 'zero_point', 'min', 'max']
        for key in key_names:
            value = getattr(self, key)
            value.data = state_dict[prefix + key].data
            state_dict.pop(prefix + key)

    def __str__(self):
        info = 'scale: %.10f ' % self.scale
        info += 'zp: %d ' % self.zero_point
        info += 'min: %.6f ' % self.min
        info += 'max: %.6f' % self.max
        return info


class QModule(nn.Module):

    def __init__(self, qi=True, qo=True, num_bits=8):
        super(QModule, self).__init__()
        if qi:
            self.qi = QParam(num_bits=num_bits)
        if qo:
            self.qo = QParam(num_bits=num_bits)

    def freeze(self):
        pass

    def quantize_inference(self, x):
        raise NotImplementedError('quantize_inference should be implemented.')


class QAvgPooling2d(QModule):

    def __init__(self, kernel_size=3, stride=1, padding=0, qi=False,
        num_bits=None):
        super(QAvgPooling2d, self).__init__(qi=qi, num_bits=num_bits)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)
        x = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return x

    def quantize_inference(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
