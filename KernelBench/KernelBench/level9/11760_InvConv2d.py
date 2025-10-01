import torch
from torch import nn
from torch.nn import functional as F


class InvConv2d(nn.Module):

    def __init__(self, in_channel):
        """
        a flow contains the equivalent of a permutation that reverses the ordering of the channels
        replact the fixed permutation with a (learned) invertible 1x1 conv, weigth matrix is initialized as a random rotation matrix
        Note: 1x1 conv with equal number of input and output channels --> generalization of a permutation operation
        :param in_channel:
        """
        super(InvConv2d, self).__init__()
        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape
        out = F.conv2d(input, self.weight)
        logdet = height * width * torch.slogdet(self.weight.squeeze().double()
            )[1].float()
        return out, logdet

    def reverse(self, output):
        return F.conv2d(output, self.weight.squeeze().inverse().unsqueeze(2
            ).unsqueeze(3))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4}]
