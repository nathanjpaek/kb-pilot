import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.utils import _pair


class SelfAttentionConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, groups=1, bias=True):
        super(SelfAttentionConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.rel_size = out_channels // groups // 2
        self.relative_x = nn.Parameter(torch.Tensor(self.rel_size, self.
            kernel_size[1]))
        self.relative_y = nn.Parameter(torch.Tensor(out_channels // groups -
            self.rel_size, self.kernel_size[0]))
        self.weight_query = nn.Conv2d(self.in_channels, self.out_channels, 
            1, groups=self.groups, bias=False)
        self.weight_key = nn.Conv2d(self.in_channels, self.out_channels, 1,
            groups=self.groups, bias=False)
        self.weight_value = nn.Conv2d(self.in_channels, self.out_channels, 
            1, groups=self.groups, bias=False)
        self.softmax = nn.Softmax(dim=3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight_query.weight, mode='fan_out',
            nonlinearity='relu')
        init.kaiming_normal_(self.weight_key.weight, mode='fan_out',
            nonlinearity='relu')
        init.kaiming_normal_(self.weight_value.weight, mode='fan_out',
            nonlinearity='relu')
        if self.bias is not None:
            bound = 1 / math.sqrt(self.out_channels)
            init.uniform_(self.bias, -bound, bound)
        init.normal_(self.relative_x, 0, 1)
        init.normal_(self.relative_y, 0, 1)

    def forward(self, x):
        b, _c, h, w = x.size()
        kh, kw = self.kernel_size
        ph, pw = h + self.padding[0] * 2, w + self.padding[1] * 2
        fh = (ph - kh) // self.stride[0] + 1
        fw = (pw - kw) // self.stride[1] + 1
        px, py = self.padding
        x = F.pad(x, (py, py, px, px))
        vq = self.weight_query(x)
        vk = self.weight_key(x)
        vv = self.weight_value(x)
        win_q = vq[:, :, (kh - 1) // 2:ph - kh // 2:self.stride[0], (kw - 1
            ) // 2:pw - kw // 2:self.stride[1]]
        win_q_b = win_q.view(b, self.groups, -1, fh, fw)
        win_q_x, win_q_y = win_q_b.split(self.rel_size, dim=2)
        win_q_x = torch.einsum('bgxhw,xk->bhwk', (win_q_x, self.relative_x))
        win_q_y = torch.einsum('bgyhw,yk->bhwk', (win_q_y, self.relative_y))
        win_k = vk.unfold(2, kh, self.stride[0]).unfold(3, kw, self.stride[1])
        vx = (win_q.unsqueeze(4).unsqueeze(4) * win_k).sum(dim=1)
        vx = vx + win_q_x.unsqueeze(3) + win_q_y.unsqueeze(4)
        vx = self.softmax(vx.view(b, fh, fw, -1)).view(b, 1, fh, fw, kh, kw)
        win_v = vv.unfold(2, kh, self.stride[0]).unfold(3, kw, self.stride[1])
        fin_v = torch.einsum('bchwkl->bchw', (vx * win_v,))
        if self.bias is not None:
            fin_v += self.bias
        return fin_v


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
