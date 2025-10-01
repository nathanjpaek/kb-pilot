import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed


class SoftTargetCrossEntropy(nn.Module):

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
