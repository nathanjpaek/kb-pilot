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


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = 全连接层(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = 全连接层(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(gelu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
