import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed


class LabelSmoothingBCE(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothingBCE, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        smooth_target = target.clone().masked_fill(target == 1, self.confidence
            )
        smooth_target = smooth_target.masked_fill(target == 0, self.smoothing)
        return self.criterion(x, smooth_target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
