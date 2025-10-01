import torch
import torch.nn as nn
import torch.nn.init


class ResidualSequential(nn.Sequential):

    def __init__(self, *args):
        super(ResidualSequential, self).__init__(*args)

    def forward(self, x):
        out = super(ResidualSequential, self).forward(x)
        x_ = None
        if out.size(2) != x.size(2) or out.size(3) != x.size(3):
            diff2 = x.size(2) - out.size(2)
            diff3 = x.size(3) - out.size(3)
            x_ = x[:, :, diff2 / 2:out.size(2) + diff2 / 2, diff3 / 2:out.
                size(3) + diff3 / 2]
        else:
            x_ = x
        return out + x_

    def eval(self):
        None
        for m in self.modules():
            m.eval()
        exit()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
