import torch
import torch.nn as nn


class AsymmetricalFocalLoss(nn.Module):

    def __init__(self, gamma=0, zeta=0):
        super(AsymmetricalFocalLoss, self).__init__()
        self.gamma = gamma
        self.zeta = zeta

    def forward(self, pred, target):
        losses = -((1 - pred) ** self.gamma * target * torch.clamp_min(
            torch.log(pred), -100) + pred ** self.zeta * (1 - target) *
            torch.clamp_min(torch.log(1 - pred), -100))
        return torch.mean(losses)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
