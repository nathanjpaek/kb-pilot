import torch
import torch.nn as nn


class IOU_Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y):
        i = y_pred.mul(y)
        u = y_pred + y - i
        mean_iou = torch.mean(i.view(i.shape[0], -1).sum(1) / u.view(i.
            shape[0], -1).sum(1))
        iou_loss = 1 - mean_iou
        return iou_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
