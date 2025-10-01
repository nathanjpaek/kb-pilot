import torch
import torch.nn as nn


class ActNorm(nn.Module):

    def __init__(self, num_channels, scale=1.0, logscale_factor=3.0,
        batch_variance=False):
        """
        Activation normalization layer

        :param num_channels: number of channels
        :type num_channels: int
        :param scale: scale
        :type scale: float
        :param logscale_factor: factor for logscale
        :type logscale_factor: float
        :param batch_variance: use batch variance
        :type batch_variance: bool
        """
        super().__init__()
        self.num_channels = num_channels
        self.scale = scale
        self.logscale_factor = logscale_factor
        self.batch_variance = batch_variance
        self.bias_inited = False
        self.logs_inited = False
        self.register_parameter('bias', nn.Parameter(torch.zeros(1, self.
            num_channels, 1, 1)))
        self.register_parameter('logs', nn.Parameter(torch.zeros(1, self.
            num_channels, 1, 1)))

    def actnorm_center(self, x, reverse=False):
        """
        center operation of activation normalization

        :param x: input
        :type x: torch.Tensor
        :param reverse: whether to reverse bias
        :type reverse: bool
        :return: centered input
        :rtype: torch.Tensor
        """
        if not self.bias_inited:
            self.initialize_bias(x)
        if not reverse:
            return x + self.bias
        else:
            return x - self.bias

    def actnorm_scale(self, x, logdet, reverse=False):
        """
        scale operation of activation normalization

        :param x: input
        :type x: torch.Tensor
        :param logdet: log determinant
        :type logdet:
        :param reverse: whether to reverse bias
        :type reverse: bool
        :return: centered input and logdet
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        if not self.logs_inited:
            self.initialize_logs(x)
        logs = self.logs * self.logscale_factor
        if not reverse:
            x *= torch.exp(logs)
        else:
            x *= torch.exp(-logs)
        if logdet is not None:
            logdet_factor = ops.count_pixels(x)
            dlogdet = torch.sum(logs) * logdet_factor
            if reverse:
                dlogdet *= -1
            logdet += dlogdet
        return x, logdet

    def initialize_bias(self, x):
        """
        Initialize bias

        :param x: input
        :type x: torch.Tensor
        """
        if not self.training:
            return
        with torch.no_grad():
            x_mean = -1.0 * ops.reduce_mean(x, dim=[0, 2, 3], keepdim=True)
            self.bias.data.copy_(x_mean.data)
            self.bias_inited = True

    def initialize_logs(self, x):
        """
        Initialize logs

        :param x: input
        :type x: torch.Tensor
        """
        if not self.training:
            return
        with torch.no_grad():
            if self.batch_variance:
                x_var = ops.reduce_mean(x ** 2, keepdim=True)
            else:
                x_var = ops.reduce_mean(x ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(x_var) + 1e-06)
                ) / self.logscale_factor
            self.logs.data.copy_(logs.data)
            self.logs_inited = True

    def forward(self, x, logdet=None, reverse=False):
        """
        Forward activation normalization layer

        :param x: input
        :type x: torch.Tensor
        :param logdet: log determinant
        :type logdet:
        :param reverse: whether to reverse bias
        :type reverse: bool
        :return: normalized input and logdet
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        assert len(x.shape) == 4
        assert x.shape[1
            ] == self.num_channels, 'Input shape should be NxCxHxW, however channels are {} instead of {}'.format(
            x.shape[1], self.num_channels)
        assert x.device == self.bias.device and x.device == self.logs.device, 'Expect input device {} instead of {}'.format(
            self.bias.device, x.device)
        if not reverse:
            x = self.actnorm_center(x, reverse=False)
            x, logdet = self.actnorm_scale(x, logdet, reverse=False)
        else:
            x, logdet = self.actnorm_scale(x, logdet, reverse=True)
            x = self.actnorm_center(x, reverse=True)
        return x, logdet


class Conv2d(nn.Conv2d):

    @staticmethod
    def get_padding(padding_type, kernel_size, stride):
        """
        Get padding size.

        mentioned in https://github.com/pytorch/pytorch/issues/3867#issuecomment-361775080
        behaves as 'SAME' padding in TensorFlow
        independent on input size when stride is 1

        :param padding_type: type of padding in ['SAME', 'VALID']
        :type padding_type: str
        :param kernel_size: kernel size
        :type kernel_size: tuple(int) or int
        :param stride: stride
        :type stride: int
        :return: padding size
        :rtype: tuple(int)
        """
        assert padding_type in ['SAME', 'VALID'
            ], 'Unsupported padding type: {}'.format(padding_type)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        if padding_type == 'SAME':
            assert stride == 1, "'SAME' padding only supports stride=1"
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3),
        stride=1, padding_type='SAME', do_weightnorm=False, do_actnorm=True,
        dilation=1, groups=1):
        """
        Wrapper of nn.Conv2d with weight normalization and activation normalization

        :param padding_type: type of padding in ['SAME', 'VALID']
        :type padding_type: str
        :param do_weightnorm: whether to do weight normalization after convolution
        :type do_weightnorm: bool
        :param do_actnorm: whether to do activation normalization after convolution
        :type do_actnorm: bool
        """
        padding = self.get_padding(padding_type, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias=not do_actnorm)
        self.do_weight_norm = do_weightnorm
        self.do_actnorm = do_actnorm
        self.weight.data.normal_(mean=0.0, std=0.05)
        if self.do_actnorm:
            self.actnorm = ActNorm(out_channels)
        else:
            self.bias.data.zero_()

    def forward(self, x):
        """
        Forward wrapped Conv2d layer

        :param x: input
        :type x: torch.Tensor
        :return: output
        :rtype: torch.Tensor
        """
        x = super().forward(x)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3),
        stride=1, padding_type='SAME', logscale_factor=3, dilation=1,
        groups=1, bias=True):
        """
        Wrapper of nn.Conv2d with zero initialization and logs

        :param padding_type: type of padding in ['SAME', 'VALID']
        :type padding_type: str
        :param logscale_factor: factor for logscale
        :type logscale_factor: float
        """
        padding = Conv2d.get_padding(padding_type, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)
        self.logscale_factor = logscale_factor
        self.bias.data.zero_()
        self.weight.data.zero_()
        self.register_parameter('logs', nn.Parameter(torch.zeros(
            out_channels, 1, 1)))

    def forward(self, x):
        """
        Forward wrapped Conv2d layer

        :param x: input
        :type x: torch.Tensor
        :return: output
        :rtype: torch.Tensor
        """
        x = super().forward(x)
        x *= torch.exp(self.logs * self.logscale_factor)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
