import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


def logsigsoftmax_v2(logits, dim=1):
    """
    v 1与 v2 差别在于 pytorch 计算softmax时有一个中心化的过程,v1 与 v2 实质上应该等同
    """
    sigmoid_logits = logits.sigmoid().log()
    sigsoftmax_logits = logits + sigmoid_logits
    return sigsoftmax_logits.log_softmax(dim=dim)


class SigSoftmaxV2(nn.Module):

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, logits):
        return logsigsoftmax_v2(logits, dim=self.dim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
