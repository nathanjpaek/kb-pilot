import torch
import torch.nn as nn


class MaxPoolWithMask(nn.Module):
    """
    带mask矩阵的max pooling。在做max-pooling的时候不会考虑mask值为0的位置。
    """

    def __init__(self):
        super(MaxPoolWithMask, self).__init__()
        self.inf = 10000000000000.0

    def forward(self, tensor, mask, dim=1):
        """
        :param torch.FloatTensor tensor: [batch_size, seq_len, channels] 初始tensor
        :param torch.LongTensor mask: [batch_size, seq_len] 0/1的mask矩阵
        :param int dim: 需要进行max pooling的维度
        :return:
        """
        masks = mask.view(mask.size(0), mask.size(1), -1)
        masks = masks.expand(-1, -1, tensor.size(2)).float()
        return torch.max(tensor + masks.le(0.5).float() * -self.inf, dim=dim)[0
            ]


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
