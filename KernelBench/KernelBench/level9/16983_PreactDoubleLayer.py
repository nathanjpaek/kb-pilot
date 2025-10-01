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


class DoubleLayer(ClippedModule):
    """
    Implementation of the double layer, also referred to as a Basic ResNet Block.

        act2( N2( K2( act1( N1( K1(Y) )))))

    Attributes:
        conv1 (sub-module): convolution class, default is 3x3 2Dconvolution
        conv2 (sub-module):           ''
        act1  (sub-module): activation function, default is ReLU()
        act2  (sub-module):           ''
        normLayer1 (sub-module): normalization with affine bias and weight, default is no normalization
        normLayer2 (sub-module):      ''

    Typical attributes for the children:
        conv#.weight (Parameter):  dims (nChanOut,nChanIn,3,3) for default 2DConvolution from nChanIn -> nChanOut channels
        conv#.bias   (Parameter):  vector, dims (nChanIn)
        normLayer#.weight (Parameter): vector, dims (nChanOut) affine scaling
        normLayer#.bias   (Parameter): vector, dims (nChanOut) affine scaling bias
    """

    def __init__(self, vFeat, params={}):
        """
        :param vFeat: 2-item list of number of expected input channels and number of channels to return, [nChanIn,nChanOut]
        :param params: dict of possible parameters ( 'conv1' , 'conv2', 'act1' , 'act2' , 'normLayer1' , 'normLayer2' )
        """
        super().__init__()
        if type(vFeat) is not list:
            vFeat = [vFeat, vFeat]
        nChanIn = vFeat[0]
        nChanOut = vFeat[1]
        szKernel = 3
        stride = 1
        padding = 1
        self.conv1 = nn.Conv2d(in_channels=nChanIn, kernel_size=szKernel,
            out_channels=nChanOut, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=nChanIn, kernel_size=szKernel,
            out_channels=nChanOut, stride=stride, padding=padding)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        if 'conv1' in params.keys():
            self.conv1 = copy.deepcopy(params.get('conv1'))
        if 'conv2' in params.keys():
            self.conv2 = copy.deepcopy(params.get('conv2'))
        if 'act1' in params.keys():
            self.act1 = params.get('act1')
        if 'act2' in params.keys():
            self.act2 = params.get('act2')
        if 'normLayer1' in params.keys():
            self.normLayer1 = copy.deepcopy(params.get('normLayer1'))
            self.normLayer1.weight.data = torch.ones(nChanOut)
            self.normLayer1.bias.data = torch.zeros(nChanOut)
        if 'normLayer2' in params.keys():
            self.normLayer2 = copy.deepcopy(params.get('normLayer2'))
            self.normLayer2.weight.data = torch.ones(nChanOut)
            self.normLayer2.bias.data = torch.zeros(nChanOut)
        if 'conv' in params.keys():
            self.conv1 = copy.deepcopy(params.get('conv'))
            self.conv2 = copy.deepcopy(self.conv1)
        if 'act' in params.keys():
            self.act1 = params.get('act')
            self.act2 = copy.deepcopy(self.act1)
        if 'normLayer' in params.keys():
            self.normLayer1 = copy.deepcopy(params.get('normLayer'))
            self.normLayer1.weight.data = torch.ones(nChanOut)
            self.normLayer2 = copy.deepcopy(self.normLayer1)
        self.conv1.weight.data = normalInit(self.conv1.weight.data.shape)
        self.conv2.weight.data = normalInit(self.conv2.weight.data.shape)
        if self.conv1.bias is not None:
            self.conv1.bias.data *= 0
        if self.conv2.bias is not None:
            self.conv2.bias.data *= 0

    def forward(self, x):
        z = self.conv1(x)
        if hasattr(self, 'normLayer1'):
            z = self.normLayer1(z)
        z = self.act1(z)
        z = self.conv2(z)
        if hasattr(self, 'normLayer2'):
            z = self.normLayer2(z)
        z = self.act2(z)
        return z

    def weight_variance(self, other):
        """apply regularization in time"""
        value = 0
        value += regMetric(nn.utils.convert_parameters.parameters_to_vector
            (self.parameters()), nn.utils.convert_parameters.
            parameters_to_vector(other.parameters()))
        return value


class PreactDoubleLayer(DoubleLayer):
    """ pre-activated version of the DoubleLayer

        N2( act2( K2( N1( act1( K1(Y) )))))
    """

    def __init__(self, vFeat, params={}):
        super().__init__(vFeat, params=params)

    def forward(self, x):
        z = self.act1(x)
        z = self.conv1(z)
        if hasattr(self, 'normLayer1'):
            z = self.normLayer1(z)
        z = self.act2(z)
        z = self.conv2(z)
        if hasattr(self, 'normLayer2'):
            z = self.normLayer2(z)
        return z


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'vFeat': 4}]
