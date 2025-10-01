import torch
import torch.nn as nn


class localSubNet(nn.Module):

    def __init__(self, blockDepth=16, convDepth=32, scale=0.25):
        super(localSubNet, self).__init__()
        self.blockDepth = blockDepth
        self.convDepth = convDepth
        self.scale = scale
        self.net = torch.nn.Sequential()
        for i in range(self.blockDepth):
            if i != self.blockDepth - 1:
                if i == 0:
                    conv = torch.nn.Conv2d(3, self.convDepth, 3, padding=1)
                    torch.nn.init.kaiming_normal_(conv.weight)
                    torch.nn.init.zeros_(conv.bias)
                else:
                    conv = torch.nn.Conv2d(self.convDepth, self.convDepth, 
                        3, padding=1)
                    torch.nn.init.kaiming_normal_(conv.weight)
                    torch.nn.init.zeros_(conv.bias)
                self.net.add_module('conv%d' % i, conv)
                self.net.add_module('leakyRelu%d' % i, torch.nn.LeakyReLU(
                    inplace=False))
            else:
                conv = torch.nn.Conv2d(self.convDepth, 3, 3, padding=1)
                torch.nn.init.kaiming_normal_(conv.weight)
                torch.nn.init.zeros_(conv.bias)
                self.net.add_module('conv%d' % i, conv)
                self.net.add_module('tanh-out', torch.nn.Tanh())

    def forward(self, x):
        local_layer = self.net(x) * self.scale
        return local_layer


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
