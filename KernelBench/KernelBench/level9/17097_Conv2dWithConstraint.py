import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):

    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0,
            maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
