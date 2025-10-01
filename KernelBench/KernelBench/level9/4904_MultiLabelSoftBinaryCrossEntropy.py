import random
import torch
import torch.nn as nn
from random import random
import random


class MultiLabelSoftBinaryCrossEntropy(nn.Module):

    def __init__(self, smooth_factor: 'float'=0, weighted: 'bool'=True, mcb:
        'bool'=False, hp_lambda: 'int'=10, epsilon: 'float'=0.1, logits=
        True, first_class_bg=False):
        super(MultiLabelSoftBinaryCrossEntropy, self).__init__()
        self.smooth_factor = smooth_factor
        self.logits = logits
        if logits:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none' if
                weighted else 'mean')
        else:
            self.criterion = nn.BCELoss(reduction='none' if weighted else
                'mean')
        self.weighted = weighted
        self.hp_lambda = hp_lambda
        self.MCB = mcb
        self.epsilon = epsilon
        self.first_class_bg = first_class_bg

    def forward(self, y_pred: 'torch.Tensor', y_true: 'torch.Tensor'
        ) ->torch.Tensor:
        if y_pred.size() != y_true.size():
            """
            Case in which y_pred.shape == b x c+1 x h x w and y_true.shape == b x c x h x w
            """
            y_pred = y_pred[:, 1:]
        b, _c, h, w = y_true.shape
        y_true = y_true.float()
        if self.smooth_factor:
            smooth = random.uniform(0, self.smooth_factor)
            soft_targets = (1 - y_true) * smooth + y_true * (1 - smooth)
        else:
            soft_targets = y_true
        bce_loss = self.criterion(y_pred, soft_targets)
        if self.weighted and not self.MCB:
            N = h * w
            weights = y_true.sum(dim=(2, 3), keepdim=True) / N
            betas = 1 - weights
            bce_loss = y_true * bce_loss * betas + (1 - y_true
                ) * bce_loss * weights
            bce_loss = bce_loss.sum() / (b * N)
        if self.weighted and self.MCB:
            Ypos = y_true.sum(dim=(0, 2, 3), keepdim=False)
            mcb_loss = 0
            for i, k in enumerate(Ypos):
                if self.first_class_bg and i == 0:
                    tmp = (y_true[:, i] * bce_loss[:, i]).flatten(1, 2)
                    mcb_loss += torch.topk(tmp, k=self.hp_lambda * 25, dim=
                        1, sorted=False).values.mean()
                else:
                    tmp = ((1 - y_true[:, i]) * bce_loss[:, i]).flatten(1, 2)
                    topk = max(min(k * self.hp_lambda // b, (1 - y_true[:,
                        i]).sum() // b), self.hp_lambda)
                    ik = torch.topk(tmp, k=int(topk), dim=1, sorted=False
                        ).values
                    beta_k = ik.shape[1] / (k / b + ik.shape[1] + self.epsilon)
                    mcb_loss += (ik * (1 - beta_k)).mean()
                    tmp = y_true[:, i] * bce_loss[:, i]
                    mcb_loss += (tmp * beta_k).sum() / (y_true[:, i].sum() +
                        self.epsilon)
            bce_loss = mcb_loss
        return bce_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
