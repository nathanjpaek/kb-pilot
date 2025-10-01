import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch._utils
import torch.nn


class KDLoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network, NIPS2014.
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, T=1, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.t = T

    def single_kl(self, s_preds, t_preds, mask=None):
        if mask is not None:
            if mask.sum() > 0:
                p = F.log_softmax(s_preds / self.t, dim=1)[mask]
                q = F.softmax(t_preds / self.t, dim=1)[mask]
                l_kl = F.kl_div(p, q, reduce=False)
                loss = torch.sum(l_kl)
                loss = loss / mask.sum()
            else:
                loss = torch.Tensor([0])
        else:
            p = F.log_softmax(s_preds / self.t, dim=1)
            q = F.softmax(t_preds / self.t, dim=1)
            l_kl = F.kl_div(p, q, reduce=False)
            loss = l_kl.sum() / l_kl.size(0)
        return loss * self.t ** 2

    def forward(self, s_preds, t_preds, masks=None):
        if masks is not None:
            assert isinstance(masks, list) and len(masks) == len(s_preds
                ), 'masks must be consistent with preds!'
        else:
            masks = [None for _ in range(len(s_preds))]
        loss = 0
        for idx, (s, t) in enumerate(zip(s_preds, t_preds)):
            loss += self.single_kl(s, t, masks[idx])
        return loss * self.loss_weight


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
