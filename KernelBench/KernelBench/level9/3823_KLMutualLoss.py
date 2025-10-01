import torch
import torch.nn as nn


class KLMutualLoss(nn.Module):

    def __init__(self):
        super(KLMutualLoss, self).__init__()
        self.kl_loss = nn.KLDivLoss(size_average=False)
        self.log_softmax = nn.functional.log_softmax
        self.softmax = nn.functional.softmax

    def forward(self, pred1, pred2):
        pred1 = self.log_softmax(pred1, dim=1)
        pred2 = self.softmax(pred2, dim=1)
        loss = self.kl_loss(pred1, pred2.detach())
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
