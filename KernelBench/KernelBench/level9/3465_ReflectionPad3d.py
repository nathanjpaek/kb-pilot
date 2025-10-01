import torch
import torch.utils.data
import torch
import torch.nn as nn


class ReflectionPad3d(nn.Module):

    def __init__(self, padding):
        super(ReflectionPad3d, self).__init__()
        self.padding = padding
        if isinstance(padding, int):
            self.padding = (padding,) * 6

    def forward(self, input):
        """
        Arguments
            :param input: tensor of shape :math:`(N, C_{	ext{in}}, H, [W, D]))`
        Returns
            :return: tensor of shape :math:`(N, C_{	ext{in}}, [D + 2 * self.padding[0],
                     H + 2 * self.padding[1]], W + 2 * self.padding[2]))`
        """
        input = torch.cat([input, input.flip([2])[:, :, 0:self.padding[-1]]
            ], dim=2)
        input = torch.cat([input.flip([2])[:, :, -self.padding[-2]:], input
            ], dim=2)
        if len(self.padding) > 2:
            input = torch.cat([input, input.flip([3])[:, :, :, 0:self.
                padding[-3]]], dim=3)
            input = torch.cat([input.flip([3])[:, :, :, -self.padding[-4]:],
                input], dim=3)
        if len(self.padding) > 4:
            input = torch.cat([input, input.flip([4])[:, :, :, :, 0:self.
                padding[-5]]], dim=4)
            input = torch.cat([input.flip([4])[:, :, :, :, -self.padding[-6
                ]:], input], dim=4)
        return input


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'padding': 4}]
