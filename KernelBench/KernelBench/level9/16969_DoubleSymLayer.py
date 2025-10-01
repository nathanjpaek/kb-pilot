import copy
import math
import torch
import torch.nn as nn


def normalInit(dims):
    """
    Essentially, PyTorch's init.xavier_normal_ but clamped
    :param K: tensor to be initialized/overwritten
    :return:  initialized tensor on the device in the nn.Parameter wrapper
    """
    K = torch.zeros(dims)
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(K)
    sd = math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        K = K.normal_(0, sd)
    K = torch.clamp(K, min=-2 * sd, max=2 * sd)
    return K


def regMetric(x, y):
    return torch.norm(x - y, p=1)


class ClippedModule(nn.Module):
    """
    Extend nn.Module to include max and min values for bound constraints / clipping
    """

    def __init__(self):
        super().__init__()
        self.minConv = -0.5
        self.maxConv = 0.5
        self.minDef = -1.5
        self.maxDef = 1.5

    def setClipValues(self, minConv=-0.5, maxConv=0.5, minDef=-1.5, maxDef=1.5
        ):
        """
            set box constraints
        :param minConv: float, lower bound for convolutions
        :param maxConv: float, upper bound for convolutions
        :param minDef:  float, lower bound for all other parameters
        :param maxDef:  float, upper bound for all other parameters
        """
        self.minConv = minConv
        self.maxConv = maxConv
        self.minDef = minDef
        self.maxDef = maxDef

    def calcClipValues(self, h, nPixels, nChan):
        """ calculation for setting bound constraints....not tuned yet"""
        mult = 1 / h
        mult = mult / math.sqrt(nPixels)
        mult = mult * (500 / nChan ** 2)
        minConv = -1
        maxConv = 1
        self.setClipValues(minConv=mult * minConv, maxConv=mult * maxConv,
            minDef=-1.5, maxDef=1.5)

    def clip(self):
        """project values onto box constraints"""
        if hasattr(self, 'conv'):
            self.conv.weight.data.clamp_(min=self.minConv, max=self.maxConv)
            if self.conv.bias is not None:
                self.conv.bias.data.clamp_(min=self.minConv, max=self.maxConv)
        if hasattr(self, 'conv1'):
            self.conv1.weight.data.clamp_(min=self.minConv, max=self.maxConv)
            if self.conv1.bias is not None:
                self.conv1.bias.data.clamp_(min=self.minConv, max=self.maxConv)
        if hasattr(self, 'conv2'):
            self.conv2.weight.data.clamp_(min=self.minConv, max=self.maxConv)
            if self.conv2.bias is not None:
                self.conv2.bias.data.clamp_(min=self.minConv, max=self.maxConv)
        if hasattr(self, 'weight'):
            w = self.weight.data
            w.clamp_(min=self.minDef, max=self.maxDef)
        for module in self.children():
            if hasattr(module, 'clip'):
                module.clip()
            else:
                if hasattr(module, 'weight'):
                    w = module.weight.data
                    w.clamp_(min=self.minDef, max=self.maxDef)
                for child in module.children():
                    if hasattr(child, 'clip'):
                        child.clip()
        return self


class DoubleSymLayer(ClippedModule):
    """
    Implementation of the double symmetric layer, also referred to as a Parabolic Layer.

        - K^T ( act( N( K(Y))))

    Attributes:
        conv (sub-module): convolution class, default is 3x3 2Dconvolution
        act  (sub-module): activation function, default is ReLU()
        normLayer (sub-module): normalization with affine bias and weight, default is no normalization

    Typical attributes for the children:
        conv.weight (Parameter):  dims (nChanOut,nChanIn,3,3) for default 2DConvolution from nChanIn -> nChanOut channels
        conv.bias   (Parameter):  vector, dims (nChanIn)
        normLayer.weight (Parameter): vector, dims (nChanOut) affine scaling
        normLayer.bias   (Parameter): vector, dims (nChanOut) affine scaling bias
    """

    def __init__(self, vFeat, params={}):
        super().__init__()
        if type(vFeat) is not list:
            vFeat = [vFeat, vFeat]
        nChanIn = vFeat[0]
        nChanOut = vFeat[1]
        szKernel = 3
        stride = 1
        padding = 1
        self.conv = nn.Conv2d(in_channels=nChanIn, kernel_size=szKernel,
            out_channels=nChanOut, stride=stride, padding=padding)
        self.act = nn.ReLU()
        if 'conv' in params.keys():
            self.conv = copy.deepcopy(params.get('conv'))
            szKernel = self.conv.kernel_size[0]
            stride = self.conv.stride
            padding = self.conv.padding
        if 'szKernel' in params.keys():
            szKernel = params.get('szKernel')
        if 'act' in params.keys():
            self.act = params.get('act')
        if 'normLayer' in params.keys():
            self.normLayer = copy.deepcopy(params.get('normLayer'))
            self.normLayer.weight.data = torch.ones(nChanOut)
        self.convt = nn.ConvTranspose2d(in_channels=nChanOut, kernel_size=
            szKernel, out_channels=nChanIn, stride=stride, padding=padding)
        self.weight = nn.Parameter(normalInit([vFeat[1], vFeat[0], szKernel,
            szKernel]), requires_grad=True)
        self.conv.weight = self.weight
        self.convt.weight = self.weight
        if self.conv.bias is not None:
            self.conv.bias.data *= 0
        if self.convt.bias is not None:
            self.convt.bias.data *= 0

    def forward(self, x):
        z = self.conv(x)
        if hasattr(self, 'normLayer'):
            z = self.normLayer(z)
        z = self.act(z)
        z = -self.convt(z)
        return z

    def calcClipValues(self, h, nPixels, nChan):
        """DoubleSym should have bound constraints half of those in DoubleLayer"""
        super().calcClipValues(h, nPixels, nChan)
        self.minConv = 0.5 * self.minConv
        self.maxConv = 0.5 * self.maxConv

    def weight_variance(self, other):
        """apply regularization in time"""
        value = 0
        value += regMetric(nn.utils.convert_parameters.parameters_to_vector
            (self.parameters()), nn.utils.convert_parameters.
            parameters_to_vector(other.parameters()))
        return value


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'vFeat': 4}]
