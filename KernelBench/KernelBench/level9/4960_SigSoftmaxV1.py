import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


def logsigsoftmax_v1(logits, dim=1):
    """
    Computes sigsoftmax from the paper - https://arxiv.org/pdf/1805.10829.pdf
    """
    max_values = torch.max(logits, dim, keepdim=True)[0]
    exp_logits_sigmoided = torch.exp(logits - max_values) * torch.sigmoid(
        logits)
    sum_exp_logits_sigmoided = exp_logits_sigmoided.sum(1, keepdim=True)
    log_probs = logits - max_values + torch.log(torch.sigmoid(logits)
        ) - torch.log(sum_exp_logits_sigmoided)
    return log_probs


class SigSoftmaxV1(nn.Module):
    """
        Sigmoid 加上 softmax的实现
    """

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, logits):
        return logsigsoftmax_v1(logits, dim=self.dim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
