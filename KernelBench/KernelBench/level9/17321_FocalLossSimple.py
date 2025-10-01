import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class FocalLossSimple(nn.Module):

    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logit, target, epoch=0):
        target = target.float()
        pred_prob = F.sigmoid(logit)
        ce = F.binary_cross_entropy_with_logits(logit, target, reduction='none'
            )
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        if self.alpha is not None:
            alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha
                )
        else:
            alpha_factor = 1
        loss = alpha_factor * modulating_factor * ce
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
