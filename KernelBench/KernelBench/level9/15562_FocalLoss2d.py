import torch
from torch import nn


class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, outputs: 'torch.Tensor', targets: 'torch.Tensor'):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        eps = 1e-08
        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1.0 - eps)
        targets = torch.clamp(targets, eps, 1.0 - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1.0 - pt) ** self.gamma * torch.log(pt)).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
