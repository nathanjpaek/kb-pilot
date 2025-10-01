import torch
import torch.nn as nn


class PrimaryCaps(nn.Module):
    """
        输入:(B,C,H,W)=(B,256,20,20)
        输出：(B,C_N,C_L)=(B,32*6*6, 8)=(B,1152,8)
        C_N:capsule_num，胶囊的个数
        C_L:capsule_length，每个胶囊的长度
    """

    def __init__(self, capsule_length=8, in_channels=256, out_channels=32,
        capsule_num=32 * 6 * 6, kernel_size=9, stride=2, padding=0):
        super(PrimaryCaps, self).__init__()
        self.capsule_length = capsule_length
        self.capsule_num = capsule_num
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels * capsule_length, kernel_size=kernel_size, stride=
            stride, padding=padding)

    def forward(self, x):
        """
        :param x: (B,C,H,W) -> (B,256,20,20)
        :return: (B,C_N,C_L) -> (100,32*6*6,8) = (100,1152,8)
        """
        x = self.conv(x)
        x = self.toCapsules(x)
        return x

    def toCapsules(self, x):
        B = x.size(0)
        x.size(1)
        H = x.size(2)
        W = x.size(3)
        x = x.reshape(B, self.capsule_length, -1, H, W)
        x = x.reshape(B, self.capsule_length, -1)
        x = self.squash(x)
        x = x.permute(0, 2, 1)
        return x

    def squash(self, input_tensor):
        """
        input_tensor: (B, 1, 10, 16)
        return: output_tensor: (B, 1, 10, 16)
        """
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1.0 + squared_norm) *
            torch.sqrt(squared_norm))
        return output_tensor


def get_inputs():
    return [torch.rand([4, 256, 64, 64])]


def get_init_inputs():
    return [[], {}]
