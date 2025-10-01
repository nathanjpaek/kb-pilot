import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo


class NonLocalBlock2D(nn.Module):

    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.
            inter_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=
            self.in_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=
            self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=
            self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'inter_channels': 4}]
