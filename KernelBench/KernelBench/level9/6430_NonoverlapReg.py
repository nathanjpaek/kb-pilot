import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel


class NonoverlapReg(nn.Module):
    """Regularization to prevent overlapping prediction of pre- and post-synaptic
    masks in synaptic polarity prediction ("1" in MODEL.TARGET_OPT).

    Args:
        fg_masked (bool): mask the regularization region with predicted cleft. Defaults: True
    """

    def __init__(self, fg_masked: 'bool'=True) ->None:
        super().__init__()
        self.fg_masked = fg_masked

    def forward(self, pred: 'torch.Tensor'):
        pos = torch.sigmoid(pred[:, 0])
        neg = torch.sigmoid(pred[:, 1])
        loss = pos * neg
        if self.fg_masked:
            loss = loss * torch.sigmoid(pred[:, 2].detach())
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
