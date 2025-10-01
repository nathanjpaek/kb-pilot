import math
import torch
import torch.nn as nn


class OhemCELoss2D(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""

    def __init__(self, n_min, thresh=0.7, ignore_index=-1):
        super(OhemCELoss2D, self).__init__(None, None, ignore_index,
            reduction='none')
        self.thresh = -math.log(thresh)
        self.n_min = n_min
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        return self.OhemCELoss(pred, target)

    def OhemCELoss(self, logits, labels):
        loss = super(OhemCELoss2D, self).forward(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_min': 4}]
