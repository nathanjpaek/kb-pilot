import torch
import torch.nn.functional as F
import torch.nn as nn


class GridAttentionBlock(nn.Module):

    def __init__(self, in_channels):
        super(GridAttentionBlock, self).__init__()
        self.inter_channels = in_channels
        self.in_channels = in_channels
        self.gating_channels = in_channels
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=
            self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.gating_channels, out_channels
            =self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True
            )
        self.psi = nn.Conv2d(in_channels=self.inter_channels, out_channels=
            1, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=
            'bilinear')
        f = F.relu(theta_x + phi_g, inplace=True)
        sigm_psi_f = torch.sigmoid(self.psi(f))
        return sigm_psi_f


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
