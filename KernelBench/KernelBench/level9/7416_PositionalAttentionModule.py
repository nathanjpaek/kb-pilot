import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalAttentionModule(nn.Module):

    def __init__(self, in_channels):
        super(PositionalAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.conv_B = nn.Conv2d(in_channels=self.in_channels, out_channels=
            self.in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_C = nn.Conv2d(in_channels=self.in_channels, out_channels=
            self.in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_D = nn.Conv2d(in_channels=self.in_channels, out_channels=
            self.in_channels, kernel_size=1, stride=1, padding=0)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, A):
        batchsize, num_channels, height, width = A.shape
        N = height * width
        B = self.conv_B(A).view((batchsize, num_channels, N))
        C = self.conv_C(A).view((batchsize, num_channels, N))
        D = self.conv_D(A).view((batchsize, num_channels, N))
        S = F.softmax(torch.bmm(C.permute(0, 2, 1), B), dim=-1)
        DS = torch.bmm(D, S.permute(0, 2, 1)).view((batchsize, num_channels,
            height, width))
        E = self.alpha * DS + A
        return E

    def initialize_weights(self):
        for layer in [self.conv_B, self.conv_C, self.conv_D]:
            nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)
            nn.init.constant_(layer.bias.data, 0.0)
        nn.init.constant_(self.alpha.data, 0.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
