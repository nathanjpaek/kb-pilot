import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch._utils
import torch.nn


class CELoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network, NIPS2014.
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, T=1, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.t = T

    def forward(self, s_preds, t_preds, **kwargs):
        loss = 0
        for s_pred, t_pred in zip(s_preds, t_preds):
            s = F.log_softmax(s_pred / self.t, dim=1)
            t = F.softmax(t_pred / self.t, dim=1)
            loss += torch.mean(torch.sum(-t * s, 1))
        return loss * self.loss_weight


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
