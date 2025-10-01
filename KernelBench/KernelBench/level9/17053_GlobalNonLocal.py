import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalNonLocal(nn.Module):

    def __init__(self, in_channel=64):
        super(GlobalNonLocal, self).__init__()
        assert in_channel % 2 == 0
        self.hidden_channel = in_channel // 2
        self.theta = nn.Conv2d(in_channel, self.hidden_channel, kernel_size
            =1, stride=1)
        self.phi = nn.Conv2d(in_channel, self.hidden_channel, kernel_size=1,
            stride=1)
        self.g = nn.Conv2d(in_channel, self.hidden_channel, kernel_size=1,
            stride=1)
        self.final = nn.Conv2d(self.hidden_channel, in_channel, kernel_size
            =1, stride=1)

    def forward(self, x):
        b, c, h, w = x.shape
        theta = self.theta(x).reshape(b, c // 2, h * w).permute(0, 2, 1
            ).contiguous()
        phi = self.phi(x).reshape(b, c // 2, h * w)
        g = self.g(x).reshape(b, c // 2, h * w).permute(0, 2, 1).contiguous()
        theta_phi = F.softmax(torch.matmul(theta, phi), dim=-1)
        theta_phi_g = torch.matmul(theta_phi, g)
        theta_phi_g = theta_phi_g.permute(0, 2, 1).contiguous().reshape(b, 
            c // 2, h, w)
        theta_phi_g = self.final(theta_phi_g)
        output = theta_phi_g + x
        return output


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
