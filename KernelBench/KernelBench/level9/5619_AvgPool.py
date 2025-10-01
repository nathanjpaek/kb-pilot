import torch
import torch.nn as nn


class AvgPool(nn.Module):
    """
    给定形如[batch_size, max_len, hidden_size]的输入，在最后一维进行avg pooling. 输出为[batch_size, hidden_size]
    """

    def __init__(self, stride=None, padding=0):
        super(AvgPool, self).__init__()
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """
        :param torch.Tensor x: [N, C, L] 初始tensor
        :return: torch.Tensor x: [N, C] avg pool后的结果
        """
        kernel_size = x.size(2)
        pooling = nn.AvgPool1d(kernel_size=kernel_size, stride=self.stride,
            padding=self.padding)
        x = pooling(x)
        return x.squeeze(dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
