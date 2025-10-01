import torch
import torch.nn as nn
import torch.utils.model_zoo


class CGD(nn.Module):

    def __init__(self, in_channels, bias=True, nonlinear=True):
        super(CGD, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.w0 = nn.Parameter(torch.ones(in_channels, 1), requires_grad=True)
        self.w1 = nn.Parameter(torch.ones(in_channels, 1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(in_channels, 1), requires_grad=True)
        self.bias0 = nn.Parameter(torch.zeros(1, in_channels, 1, 1),
            requires_grad=True)
        self.bias1 = nn.Parameter(torch.zeros(1, in_channels, 1, 1),
            requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(1, in_channels, 1, 1),
            requires_grad=True)
        nn.init.xavier_uniform_(self.w0)
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x):
        b, c, _, _ = x.size()
        x0 = self.avg_pool(x).view(b, c, 1, 1)
        x1 = self.max_pool(x).view(b, c, 1, 1)
        x0_s = self.softmax(x0)
        y0 = torch.matmul(x0.view(b, c), self.w0).view(b, 1, 1, 1)
        y1 = torch.matmul(x1.view(b, c), self.w1).view(b, 1, 1, 1)
        y0_s = torch.tanh(y0 * x0_s + self.bias0)
        y1_s = torch.tanh(y1 * x0_s + self.bias1)
        y2 = torch.matmul(y1_s.view(b, c), self.w2).view(b, 1, 1, 1)
        y2_s = torch.tanh(y2 * y0_s + self.bias2).view(b, c, 1, 1)
        z = x * (y2_s + 1)
        return z


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
