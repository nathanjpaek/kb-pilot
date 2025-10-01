import torch
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    """SoftDice loss

    """

    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
        self.SM = nn.Sigmoid()

    def forward(self, logits, labels):
        num = labels.size(0)
        m1 = self.SM(logits).view(num, -1)
        m2 = labels.view(num, -1)
        intersection = m1 * m2
        score = 2.0 * (intersection.sum(1) + 1e-15) / (m1.sum(1) + m2.sum(1
            ) + 1e-15)
        score = 1 - score.sum() / num
        return score


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
