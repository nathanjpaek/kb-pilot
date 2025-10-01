import torch
from torch import nn


class GeneratorLoss(nn.Module):

    def __init__(self, alpha=1, beta=10, gamma=10):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, fake_pred, iden, cycle, real):
        fake_target = torch.ones_like(fake_pred)
        adv_loss = self.bce(fake_pred, fake_target)
        iden_loss = self.l1(iden, real)
        cycle_loss = self.l1(cycle, real)
        loss = (self.alpha * adv_loss + self.beta * iden_loss + self.gamma *
            cycle_loss)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
