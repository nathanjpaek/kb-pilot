import math
import torch
import torch.nn as nn
import torch.utils.data.distributed


class reduction_1x1(nn.Sequential):

    def __init__(self, num_in_filters, num_out_filters, max_depth, is_final
        =False):
        super(reduction_1x1, self).__init__()
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()
        while num_out_filters >= 4:
            if num_out_filters < 8:
                if self.is_final:
                    self.reduc.add_module('final', torch.nn.Sequential(nn.
                        Conv2d(num_in_filters, out_channels=1, bias=False,
                        kernel_size=1, stride=1, padding=0), nn.Sigmoid()))
                else:
                    self.reduc.add_module('plane_params', torch.nn.Conv2d(
                        num_in_filters, out_channels=3, bias=False,
                        kernel_size=1, stride=1, padding=0))
                break
            else:
                self.reduc.add_module('inter_{}_{}'.format(num_in_filters,
                    num_out_filters), torch.nn.Sequential(nn.Conv2d(
                    in_channels=num_in_filters, out_channels=
                    num_out_filters, bias=False, kernel_size=1, stride=1,
                    padding=0), nn.ELU()))
            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2

    def forward(self, net):
        net = self.reduc.forward(net)
        if not self.is_final:
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = torch.cat([n1, n2, n3, n4], dim=1)
        return net


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_in_filters': 4, 'num_out_filters': 4, 'max_depth': 1}]
