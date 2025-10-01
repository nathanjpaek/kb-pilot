import math
import torch
import numpy as np
import torch.nn as nn


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 *
        torch.pow(x, 3))))


class 全连接层(nn.Module):

    def __init__(self, 输入_接口, 输出_接口):
        super().__init__()
        np.random.seed(1)
        self.weight = nn.Parameter(torch.FloatTensor(np.random.uniform(-1 /
            np.sqrt(输入_接口), 1 / np.sqrt(输入_接口), (输入_接口, 输出_接口))))
        self.bias = nn.Parameter(torch.FloatTensor(np.random.uniform(-1 /
            np.sqrt(输入_接口), 1 / np.sqrt(输入_接口), 输出_接口)))

    def forward(self, x):
        输出 = torch.matmul(x, self.weight)
        输出 = 输出 + self.bias
        return 输出


class 判断状态(nn.Module):

    def __init__(self, 种类数, 隐藏层尺寸, 输入层尺寸=2048, 输入尺寸A=36):
        super().__init__()
        self.隐藏层尺寸 = 隐藏层尺寸
        self.输入层尺寸 = 输入层尺寸
        self.输入尺寸A = 输入尺寸A
        self.输入层 = 全连接层(输入层尺寸, 隐藏层尺寸)
        self.隐藏层 = 全连接层(隐藏层尺寸, 隐藏层尺寸)
        self.输出层 = 全连接层(隐藏层尺寸 * 输入尺寸A, 种类数)

    def forward(self, 图向量):
        图向量 = 图向量.reshape((图向量.shape[0], self.输入尺寸A, self.输入层尺寸))
        中间量 = gelu(self.输入层(图向量))
        中间量 = self.隐藏层(中间量)
        中间量 = 中间量.reshape((中间量.shape[0], self.隐藏层尺寸 * self.输入尺寸A))
        结果 = self.输出层(中间量)
        return 结果


def get_inputs():
    return [torch.rand([4, 36, 2048])]


def get_init_inputs():
    return [[], {'种类数': 4, '隐藏层尺寸': 4}]
