import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch._utils
import torch.nn


class KLLoss(nn.Module):
    """
    KL Divergence loss
    """

    def __init__(self, norm='softmax', loss_weight=1.0):
        super(KLLoss, self).__init__()
        self.loss_weight = loss_weight
        self.norm = norm

    def forward(self, s_features, t_features, **kwargs):
        loss = 0
        for s, t in zip(s_features, t_features):
            loss += self.kl(s, t)
        return loss * self.loss_weight

    def kl(self, pred_feas, target_feas):
        crit = nn.KLDivLoss(reduction='batchmean')
        relu = nn.ReLU()
        s = relu(pred_feas)
        t = relu(target_feas)
        if self.norm == 'softmax':
            s = F.log_softmax(s, dim=1)
            t = F.softmax(t, dim=1)
            t.detach_()
            loss = crit(s, t)
        elif self.norm == 'l2':
            loss = torch.sum(t / torch.sum(t) * torch.log((t / torch.sum(t) +
                1e-06) / (s / torch.sum(s) + 1e-06)))
        else:
            None
            return None
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
