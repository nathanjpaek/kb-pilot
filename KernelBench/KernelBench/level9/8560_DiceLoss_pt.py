import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss_pt(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_pt, self).__init__()

    def forward(self, y_pred, y_true):
        smooth = 1.0
        y_pred_sig = F.sigmoid(y_pred)
        num = y_true.size(0)
        x = y_pred_sig.view(num, -1).float()
        y = y_true.view(num, -1).float()
        intersection = torch.sum(x * y)
        score = (2.0 * intersection + smooth) / (torch.sum(x) + torch.sum(y
            ) + smooth)
        out = 1 - torch.sum(score) / num
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
