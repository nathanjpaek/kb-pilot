import torch
import torch.nn as nn


class HingeMarginLoss(nn.Module):
    """
    计算hinge loss 接口
    """

    def __init__(self):
        super(HingeMarginLoss, self).__init__()

    def forward(self, t, tr, delt=None, size_average=False):
        """
        计算hingle loss
        """
        if delt is None:
            loss = torch.clamp(1 - t + tr, min=0)
        else:
            loss = torch.clamp(1 - torch.mul(t - tr, torch.squeeze(delt)),
                min=0)
            loss = torch.unsqueeze(loss, dim=-1)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
