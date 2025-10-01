import torch
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F


def Normalizations(tensor_size=None, normalization=None, available=False,
    **kwargs):
    """Does normalization on 4D tensor.

    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        normalization: None/batch/group/instance/layer/pixelwise
        available: if True, returns all available normalization methods
        groups: for group (GroupNorm), when not provided groups is the center
            value of all possible - ex: for a tensor_size[1] = 128, groups is
            set to 16 as the possible groups are [1, 2, 4, 8, 16, 32, 64, 128]
        affine: for group and instance normalization, default False
        elementwise_affine: for layer normalization. default True
    """
    list_available = ['batch', 'group', 'instance', 'layer', 'pixelwise']
    if available:
        return list_available
    normalization = normalization.lower()
    assert normalization in list_available, 'Normalization must be None/' + '/'.join(
        list_available)
    if normalization == 'batch':
        return torch.nn.BatchNorm2d(tensor_size[1])
    elif normalization == 'group':
        affine = kwargs['affine'] if 'affine' in kwargs.keys() else False
        if 'groups' in kwargs.keys():
            return torch.nn.GroupNorm(kwargs['groups'], tensor_size[1],
                affine=affine)
        else:
            possible = [(tensor_size[1] // i) for i in range(tensor_size[1],
                0, -1) if tensor_size[1] % i == 0]
            groups = possible[len(possible) // 2]
            return torch.nn.GroupNorm(groups, tensor_size[1], affine=affine)
    elif normalization == 'instance':
        affine = kwargs['affine'] if 'affine' in kwargs.keys() else False
        return torch.nn.InstanceNorm2d(tensor_size[1], affine=affine)
    elif normalization == 'layer':
        elementwise_affine = kwargs['elementwise_affine'
            ] if 'elementwise_affine' in kwargs.keys() else True
        return torch.nn.LayerNorm(tensor_size[1:], elementwise_affine=
            elementwise_affine)
    elif normalization == 'pixelwise':
        return PixelWise()


class Activations(nn.Module):
    """ All the usual activations along with maxout, relu + maxout and swish.
    MaxOut (maxo) - https://arxiv.org/pdf/1302.4389.pdf
    Swish - https://arxiv.org/pdf/1710.05941v1.pdf

    Args:
        activation: relu/relu6/lklu/elu/prelu/tanh/sigm/maxo/rmxo/swish
        channels: parameter for prelu, default is 1
    """

    def __init__(self, activation='relu', channels=1):
        super(Activations, self).__init__()
        if activation is not None:
            activation = activation.lower()
        self.activation = activation
        self.function = None
        if activation in self.available():
            self.function = getattr(self, '_' + activation)
            if activation == 'prelu':
                self.weight = nn.Parameter(torch.rand(channels))
        else:
            self.activation = ''

    def forward(self, tensor):
        if self.function is None:
            return tensor
        return self.function(tensor)

    def _relu(self, tensor):
        return F.relu(tensor)

    def _relu6(self, tensor):
        return F.relu6(tensor)

    def _lklu(self, tensor):
        return F.leaky_relu(tensor)

    def _elu(self, tensor):
        return F.elu(tensor)

    def _prelu(self, tensor):
        return F.prelu(tensor, self.weight)

    def _tanh(self, tensor):
        return torch.tanh(tensor)

    def _sigm(self, tensor):
        return torch.sigmoid(tensor)

    def _maxo(self, tensor):
        assert tensor.size(1) % 2 == 0, 'MaxOut: tensor.size(1) must be even'
        return torch.max(*tensor.split(tensor.size(1) // 2, 1))

    def _rmxo(self, tensor):
        return self._maxo(F.relu(tensor))

    def _swish(self, tensor):
        return tensor * torch.sigmoid(tensor)

    def __repr__(self):
        return self.activation

    @staticmethod
    def available():
        return ['relu', 'relu6', 'lklu', 'elu', 'prelu', 'tanh', 'sigm',
            'maxo', 'rmxo', 'swish']


class PixelWise(torch.nn.Module):
    """ Implemented - https://arxiv.org/pdf/1710.10196.pdf """

    def __init__(self, eps=1e-08):
        super(PixelWise, self).__init__()
        self.eps = eps

    def forward(self, tensor):
        return tensor.div(tensor.pow(2).mean(1, True).add(self.eps).pow(0.5))

    def __repr__(self):
        return 'pixelwise'


class ConvolutionTranspose(nn.Module):
    """
        Parameters/Inputs
            tensor_size = (None/any integer >0, channels, height, width)
            filter_size = list(length=2)/tuple(length=2)/integer
            out_channels = return tensor.size(1)
            strides = list(length=2)/tuple(length=2)/integer
            pad = True/False
            activation = "relu"/"relu6"/"lklu"/"tanh"/"sigm"/"maxo"/"rmxo"/"swish"
            dropout = 0.-1.
            normalization = None/"batch"/"group"/"instance"/"layer"/"pixelwise"
            pre_nm = True/False
            groups = 1, ... out_channels
            weight_nm = True/False -- https://arxiv.org/pdf/1602.07868.pdf
            equalized = True/False -- https://arxiv.org/pdf/1710.10196.pdf
    """

    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 
        1), pad=True, activation='relu', dropout=0.0, normalization=None,
        pre_nm=False, groups=1, weight_nm=False, equalized=False, **kwargs):
        super(ConvolutionTranspose, self).__init__()
        assert len(tensor_size) == 4 and type(tensor_size) in [list, tuple
            ], 'ConvolutionTranspose -- tensor_size must be of length 4 (tuple or list)'
        assert type(filter_size) in [int, list, tuple
            ], 'Convolution -- filter_size must be int/tuple/list'
        if isinstance(filter_size, int):
            filter_size = filter_size, filter_size
        if isinstance(filter_size, list):
            filter_size = tuple(filter_size)
        assert len(filter_size
            ) == 2, 'ConvolutionTranspose -- filter_size length must be 2'
        assert type(strides) in [int, list, tuple
            ], 'ConvolutionTranspose -- strides must be int/tuple/list'
        if isinstance(strides, int):
            strides = strides, strides
        if isinstance(strides, list):
            strides = tuple(strides)
        assert len(strides
            ) == 2, 'ConvolutionTranspose -- strides length must be 2'
        assert isinstance(pad, bool
            ), 'ConvolutionTranspose -- pad must be boolean'
        assert isinstance(dropout, float
            ), 'ConvolutionTranspose -- dropout must be float'
        assert normalization in [None, 'batch', 'group', 'instance',
            'layer', 'pixelwise'
            ], "Convolution's normalization must be None/batch/group/instance/layer/pixelwise"
        assert isinstance(equalized, bool
            ), 'Convolution -- equalized must be boolean'
        self.equalized = equalized
        if activation is not None:
            activation = activation.lower()
        dilation = kwargs['dilation'] if 'dilation' in kwargs.keys() else (1, 1
            )
        padding = (filter_size[0] // 2, filter_size[1] // 2) if pad else (0, 0)
        if dropout > 0.0:
            self.dropout = nn.Dropout2d(dropout)
        pre_expansion, pst_expansion = 1, 1
        if activation in ('maxo', 'rmxo'):
            if pre_nm:
                pre_expansion = 2
            if not pre_nm:
                pst_expansion = 2
        if pre_nm:
            if normalization is not None:
                self.Normalization = Normalizations(tensor_size,
                    normalization, **kwargs)
            if activation in ['relu', 'relu6', 'lklu', 'tanh', 'sigm',
                'maxo', 'rmxo', 'swish']:
                self.Activation = Activations(activation)
        if weight_nm:
            self.ConvolutionTranspose = nn.utils.weight_norm(nn.
                ConvTranspose2d(tensor_size[1] // pre_expansion, 
                out_channels * pst_expansion, filter_size, strides, padding,
                bias=False, dilation=dilation, groups=groups), name='weight')
        else:
            self.ConvolutionTranspose = nn.ConvTranspose2d(tensor_size[1] //
                pre_expansion, out_channels * pst_expansion, filter_size,
                strides, padding, bias=False, groups=groups)
            nn.init.kaiming_normal_(self.ConvolutionTranspose.weight, nn.
                init.calculate_gain('conv2d'))
            if equalized:
                import numpy as np
                gain = kwargs['gain'] if 'gain' in kwargs.keys() else np.sqrt(2
                    )
                fan_in = tensor_size[1] * out_channels * filter_size[0]
                self.scale = gain / np.sqrt(fan_in)
                self.ConvolutionTranspose.weight.data.mul_(self.scale)
        self.oc = self.ConvolutionTranspose.weight.data.size(0)
        self.tensor_size = tensor_size[0], out_channels, (tensor_size[2] - 1
            ) * strides[0] - 2 * padding[0] + filter_size[0], (tensor_size[
            3] - 1) * strides[1] - 2 * padding[1] + filter_size[1]
        if not pre_nm:
            if normalization is not None:
                self.Normalization = Normalizations((self.tensor_size[0], 
                    out_channels * pst_expansion, self.tensor_size[2], self
                    .tensor_size[3]), normalization, **kwargs)
            if activation in ['relu', 'relu6', 'lklu', 'tanh', 'sigm',
                'maxo', 'rmxo', 'swish']:
                self.Activation = Activations(activation)
        self.pre_nm = pre_nm

    def forward(self, tensor, output_size=None):
        if hasattr(self, 'dropout'):
            tensor = self.dropout(tensor)
        if self.pre_nm:
            if hasattr(self, 'Normalization'):
                tensor = self.Normalization(tensor)
            if hasattr(self, 'Activation'):
                tensor = self.Activation(tensor)
            if output_size is None:
                output_size = self.tensor_size
            output_size = tensor.size(0), self.oc, output_size[2], output_size[
                3]
            tensor = self.ConvolutionTranspose(tensor, output_size=output_size)
            if self.equalized:
                tensor = tensor.mul(self.scale)
        else:
            if output_size is None:
                output_size = self.tensor_size
            output_size = tensor.size(0), self.oc, output_size[2], output_size[
                3]
            tensor = self.ConvolutionTranspose(tensor, output_size=output_size)
            if self.equalized:
                tensor = tensor.mul(self.scale)
            if hasattr(self, 'Normalization'):
                tensor = self.Normalization(tensor)
            if hasattr(self, 'Activation'):
                tensor = self.Activation(tensor)
        return tensor


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'tensor_size': [4, 4, 4, 4], 'filter_size': 4,
        'out_channels': 4}]
