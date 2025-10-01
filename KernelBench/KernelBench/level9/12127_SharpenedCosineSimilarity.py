import torch
import torch.nn as nn
import torch.nn.functional as F


def unfold2d(x, kernel_size: 'int', stride: 'int', padding: 'int'):
    x = F.pad(x, [padding] * 4)
    bs, in_c, h, w = x.size()
    ks = kernel_size
    strided_x = x.as_strided((bs, in_c, (h - ks) // stride + 1, (w - ks) //
        stride + 1, ks, ks), (in_c * h * w, h * w, stride * w, stride, w, 1))
    return strided_x


class SharpenedCosineSimilarity(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, kernel_size=1, stride
        =1, padding=0, eps=1e-12):
        super(SharpenedCosineSimilarity, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.padding = int(padding)
        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        nn.init.xavier_uniform_(w)
        self.w = nn.Parameter(w.view(out_channels, in_channels, -1),
            requires_grad=True)
        self.p_scale = 10
        p_init = 2 ** 0.5 * self.p_scale
        self.register_parameter('p', nn.Parameter(torch.empty(out_channels)))
        nn.init.constant_(self.p, p_init)
        self.q_scale = 100
        self.register_parameter('q', nn.Parameter(torch.empty(1)))
        nn.init.constant_(self.q, 10)

    def forward(self, x):
        x = unfold2d(x, kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding)
        n, c, h, w, _, _ = x.shape
        x = x.reshape(n, c, h, w, -1)
        square_sum = torch.sum(torch.square(x), [1, 4], keepdim=True)
        x_norm = torch.add(torch.sqrt(square_sum + self.eps), torch.square(
            self.q / self.q_scale))
        square_sum = torch.sum(torch.square(self.w), [1, 2], keepdim=True)
        w_norm = torch.add(torch.sqrt(square_sum + self.eps), torch.square(
            self.q / self.q_scale))
        x = torch.einsum('nchwl,vcl->nvhw', x / x_norm, self.w / w_norm)
        sign = torch.sign(x)
        x = torch.abs(x) + self.eps
        x = x.pow(torch.square(self.p / self.p_scale).view(1, -1, 1, 1))
        return sign * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
