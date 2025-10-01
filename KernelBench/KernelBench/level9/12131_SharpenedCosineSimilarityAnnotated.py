import torch
import torch.nn as nn
import torch.nn.functional as F


class SharpenedCosineSimilarityAnnotated(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, kernel_size=1, stride
        =1, padding=0, eps=1e-12):
        super(SharpenedCosineSimilarityAnnotated, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.padding = int(padding)
        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        self.register_parameter('w', nn.Parameter(w))
        nn.init.xavier_uniform_(self.w)
        self.p_scale = 10
        p_init = 2 ** 0.5 * self.p_scale
        self.register_parameter('p', nn.Parameter(torch.empty(out_channels)))
        nn.init.constant_(self.p, p_init)
        self.q_scale = 100
        self.register_parameter('q', nn.Parameter(torch.empty(1)))
        nn.init.constant_(self.q, 10)

    def forward(self, x):
        w_norm = torch.linalg.vector_norm(self.w, dim=(1, 2, 3), keepdim=True)
        q_sqr = (self.q / self.q_scale) ** 2
        w_normed = self.w / (w_norm + self.eps + q_sqr)
        x_norm_squared = F.avg_pool2d((x + self.eps) ** 2, kernel_size=self
            .kernel_size, stride=self.stride, padding=self.padding,
            divisor_override=1).sum(dim=1, keepdim=True)
        y_denorm = F.conv2d(x, w_normed, bias=None, stride=self.stride,
            padding=self.padding)
        y = y_denorm / (x_norm_squared.sqrt() + q_sqr)
        sign = torch.sign(y)
        y = torch.abs(y) + self.eps
        p_sqr = (self.p / self.p_scale) ** 2
        y = y.pow(p_sqr.reshape(1, -1, 1, 1))
        return sign * y


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
