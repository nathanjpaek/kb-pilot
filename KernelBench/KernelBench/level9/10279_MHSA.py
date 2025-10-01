import torch
from torch import nn


class MHSA(nn.Module):

    def __init__(self, height, width, dim, head):
        super(MHSA, self).__init__()
        self.head = head
        self.r_h = nn.Parameter(data=torch.randn(1, head, dim // head, 1,
            height), requires_grad=True)
        self.r_w = nn.Parameter(data=torch.randn(1, head, dim // head,
            width, 1), requires_grad=True)
        self.w_q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size
            =1, stride=1, bias=True)
        self.w_k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size
            =1, stride=1, bias=True)
        self.w_v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size
            =1, stride=1, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        batch, dim, height, width = x.size()
        q = self.w_q(x).view(batch, self.head, dim // self.head, -1).permute(
            0, 1, 3, 2)
        k = self.w_k(x).view(batch, self.head, dim // self.head, -1)
        v = self.w_v(x).view(batch, self.head, dim // self.head, -1)
        r = (self.r_h + self.r_w).view(1, self.head, dim // self.head, -1)
        content_position = torch.matmul(q, r)
        content_content = torch.matmul(q, k)
        energy = (content_content + content_position).view(batch, -1)
        attention = self.softmax(energy).view(batch, self.head, height *
            width, height * width)
        feature = torch.matmul(v, attention).view(batch, dim, height, width)
        out = self.pool(feature)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'height': 4, 'width': 4, 'dim': 4, 'head': 4}]
