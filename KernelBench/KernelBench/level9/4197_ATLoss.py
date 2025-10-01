import torch
import torch.nn as nn
import torch.nn.functional as F


class ATLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        th_label = torch.zeros_like(labels, dtype=torch.float)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0
        p_mask = labels + th_label
        n_mask = 1 - labels
        logit1 = logits - (1 - p_mask) * 1e+30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        logit2 = logits - (1 - n_mask) * 1e+30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)
        loss = loss1 + loss2
        loss = loss.mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
