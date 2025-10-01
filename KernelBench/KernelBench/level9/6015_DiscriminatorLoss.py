import torch
import torch.nn as nn


class AdvLoss(nn.Module):
    """BCE for True and False reals"""

    def __init__(self, alpha=1):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        return self.alpha * self.loss_fn(pred, target)


class DiscriminatorLoss(nn.Module):
    """Discriminator loss"""

    def __init__(self, alpha=1):
        super().__init__()
        self.bce = AdvLoss(alpha)

    def forward(self, fake_pred, real_pred):
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        loss = (self.bce(fake_pred, fake_target) + self.bce(real_pred,
            real_target)) / 2
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
