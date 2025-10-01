import torch
import torch.nn as nn


class AvgPoolWithMask(nn.Module):
    """
    给定形如[batch_size, max_len, hidden_size]的输入，在最后一维进行avg pooling. 输出为[batch_size, hidden_size], pooling
    的时候只会考虑mask为1的位置
    """

    def __init__(self):
        super(AvgPoolWithMask, self).__init__()
        self.inf = 10000000000000.0

    def forward(self, tensor, mask, dim=1):
        """
        :param torch.FloatTensor tensor: [batch_size, seq_len, channels] 初始tensor
        :param torch.LongTensor mask: [batch_size, seq_len] 0/1的mask矩阵
        :param int dim: 需要进行max pooling的维度
        :return:
        """
        masks = mask.view(mask.size(0), mask.size(1), -1).float()
        return torch.sum(tensor * masks.float(), dim=dim) / torch.sum(masks
            .float(), dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 16]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
