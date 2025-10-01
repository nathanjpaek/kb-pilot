import torch
import torch.nn as nn


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        self.aspp_num = len(rates)
        for i, rate in enumerate(rates):
            self.add_module('c{}'.format(i), nn.Conv2d(in_ch, out_ch, 3, 1,
                padding=rate, dilation=rate, bias=True))
        self.add_module('bcm', nn.Conv2d(in_ch, out_ch, 1))
        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        aspp = [stage(x) for stage in self.children()]
        bcm = aspp[-1]
        return sum(aspp[:self.aspp_num]), bcm


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'out_ch': 4, 'rates': [4, 4]}]
