import torch
import torch.nn as nn


class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, eps=1e-08):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.eps = eps

    def IN_noWeight(self, x):
        N, C = x.size(0), x.size(1)
        mean = x.contiguous().view(N, C, -1).mean(2).contiguous().view(N, C,
            1, 1)
        x = x - mean
        var = torch.mul(x, x)
        var = var.contiguous().view(N, C, -1).mean(2).contiguous().view(N,
            C, 1, 1)
        var = torch.rsqrt(var + self.eps)
        x = x * var
        return x

    def Apply_style(self, content, style):
        style = style.contiguous().view([-1, 2, content.size(1), 1, 1])
        content = content * style[:, 0] + style[:, 1]
        return content

    def forward(self, content, style):
        normalized_content = self.IN_noWeight(content)
        stylized_content = self.Apply_style(normalized_content, style)
        return stylized_content


def get_inputs():
    return [torch.rand([256, 4, 4, 4]), torch.rand([32, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
