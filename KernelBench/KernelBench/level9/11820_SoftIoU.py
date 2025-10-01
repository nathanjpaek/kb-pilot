import torch
import torch.nn as nn
import torch._utils


class SoftIoU(nn.Module):

    def __init__(self, from_sigmoid=False, ignore_label=-1):
        super().__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        if not self._from_sigmoid:
            pred = torch.sigmoid(pred)
        loss = 1.0 - torch.sum(pred * label * sample_weight, dim=(1, 2, 3)) / (
            torch.sum(torch.max(pred, label) * sample_weight, dim=(1, 2, 3)
            ) + 1e-08)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
