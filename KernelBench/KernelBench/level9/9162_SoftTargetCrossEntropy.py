import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: 'torch.Tensor', target: 'torch.Tensor'
        ) ->torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
