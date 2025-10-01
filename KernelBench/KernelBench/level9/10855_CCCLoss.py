import torch
import torch.nn as nn


class CCCLoss(nn.Module):
    """CCC loss for VA regression
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def get_name(self):
        return 'CCC_loss'

    def forward(self, cls_score, reg_label, **kwargs):
        x, y = cls_score, reg_label
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.
            sqrt(torch.sum(vy ** 2)) + 1e-10)
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2 * rho * x_s * y_s / (x_s ** 2 + y_s ** 2 + (x_m - y_m) ** 2 +
            1e-10)
        loss = 1 - ccc
        return loss * self.loss_weight


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
