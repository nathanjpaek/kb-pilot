import torch
import torch.optim
from typing import Any
from typing import NoReturn
import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """ 对比损失函数"""

    def __init__(self) ->NoReturn:
        super(ContrastiveLoss, self).__init__()

    def forward(self, ew: 'Any', label: 'Any', m: 'float'):
        """
        :param ew: Embedding向量之间的度量
        :param label: 样本句子的标签
        :param m: 负样本控制阈值
        :return:
        """
        l_1 = 0.25 * (1.0 - ew) * (1.0 - ew)
        l_0 = torch.where(ew < m * torch.ones_like(ew), torch.full_like(ew,
            0), ew) * torch.where(ew < m * torch.ones_like(ew), torch.
            full_like(ew, 0), ew)
        loss = label * 1.0 * l_1 + (1 - label) * 1.0 * l_0
        return loss.sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
