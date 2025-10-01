import torch
import torch.nn as nn


def squash(inputs, axis=-1):
    """capsule输出的激活函数"""
    norm = torch.norm(inputs, dim=axis, keepdim=True)
    scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-08)
    return scale * inputs


class PrimaryCaps(nn.Module):
    """计算第一层capsules的输入，转换成32*6*6个8维的capsule vector
    in_channels：原文中256
    out_channels：卷积后的通道数，原文中256
    dim_caps: PrimaryCaps输出的每个capsule的维度
    kernel_size：原文中9 * 9
    stride：2
    """

    def __init__(self, in_channels, out_channels, dim_caps, kernel_size,
        stride=1, padding=0):
        super(PrimaryCaps, self).__init__()
        self.dim_caps = dim_caps
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=padding)

    def forward(self, input):
        """转换成32*6*6个8维的capsule vector, output size=[batch_size, num_caps, dim_caps]"""
        output = self.conv2d(input)
        output = output.view(input.size(0), -1, self.dim_caps)
        return squash(output)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'dim_caps': 4,
        'kernel_size': 4}]
