import math
import torch
from torch import nn
from torch.nn.parameter import Parameter


def weights_init(init_type='gaussian'):

    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0
            ) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, 'Unsupported initialization: {}'.format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_fun


class GaussActivation(nn.Module):

    def __init__(self, a, mu, sigma1, sigma2):
        super(GaussActivation, self).__init__()
        self.a = Parameter(torch.tensor(a, dtype=torch.float32))
        self.mu = Parameter(torch.tensor(mu, dtype=torch.float32))
        self.sigma1 = Parameter(torch.tensor(sigma1, dtype=torch.float32))
        self.sigma2 = Parameter(torch.tensor(sigma2, dtype=torch.float32))

    def forward(self, inputFeatures):
        self.a.data = torch.clamp(self.a.data, 1.01, 6.0)
        self.mu.data = torch.clamp(self.mu.data, 0.1, 3.0)
        self.sigma1.data = torch.clamp(self.sigma1.data, 1.0, 2.0)
        self.sigma2.data = torch.clamp(self.sigma2.data, 1.0, 2.0)
        lowerThanMu = inputFeatures < self.mu
        largerThanMu = inputFeatures >= self.mu
        leftValuesActiv = self.a * torch.exp(-self.sigma1 * (inputFeatures -
            self.mu) ** 2)
        leftValuesActiv.masked_fill_(largerThanMu, 0.0)
        rightValueActiv = 1 + (self.a - 1) * torch.exp(-self.sigma2 * (
            inputFeatures - self.mu) ** 2)
        rightValueActiv.masked_fill_(lowerThanMu, 0.0)
        output = leftValuesActiv + rightValueActiv
        return output


class MaskUpdate(nn.Module):

    def __init__(self, alpha):
        super(MaskUpdate, self).__init__()
        self.updateFunc = nn.ReLU(False)
        self.alpha = Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, inputMaskMap):
        self.alpha.data = torch.clamp(self.alpha.data, 0.6, 0.8)
        return torch.pow(self.updateFunc(inputMaskMap), self.alpha)


class ReverseMaskConv(nn.Module):

    def __init__(self, inputChannels, outputChannels, kernelSize=4, stride=
        2, padding=1, dilation=1, groups=1, convBias=False):
        super(ReverseMaskConv, self).__init__()
        self.reverseMaskConv = nn.Conv2d(inputChannels, outputChannels,
            kernelSize, stride, padding, dilation, groups, bias=convBias)
        self.reverseMaskConv.apply(weights_init())
        self.activationFuncG_A = GaussActivation(1.1, 2.0, 1.0, 1.0)
        self.updateMask = MaskUpdate(0.8)

    def forward(self, inputMasks):
        maskFeatures = self.reverseMaskConv(inputMasks)
        maskActiv = self.activationFuncG_A(maskFeatures)
        maskUpdate = self.updateMask(maskFeatures)
        return maskActiv, maskUpdate


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inputChannels': 4, 'outputChannels': 4}]
