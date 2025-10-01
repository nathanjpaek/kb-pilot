import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, device=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if self.alpha is not None:
            self.alpha = torch.FloatTensor([1 - alpha, alpha])

    def forward(self, pred, target):
        batch_size, n_pts = pred.size()
        pos = torch.sigmoid(pred)
        neg = 1 - pos
        pt = torch.stack([neg, pos], dim=-1).view(-1, 2)
        index = target.view(-1, 1).long()
        pt = pt.gather(-1, index).view(-1)
        logpt = pt.log()
        if self.alpha is not None:
            at = self.alpha.gather(0, index.view(-1))
            logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.view(batch_size, n_pts)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
