import torch
from torch import nn


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        if in_dim >= 8:
            self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=
                in_dim // 8, kernel_size=1, bias=False)
            self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=
                in_dim // 8, kernel_size=1, bias=False)
        else:
            self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=
                in_dim, kernel_size=1, bias=False)
            self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=
                in_dim, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,
            kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height
            ).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out, attention


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'activation': 4}]
