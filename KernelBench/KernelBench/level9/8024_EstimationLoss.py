import torch
import torch.nn as nn


class EstimationLoss(nn.Module):

    def __init__(self):
        super(EstimationLoss, self).__init__()
        self.gamma = 0
        self.alpha = 0

    def forward(self, pred, target):
        temp1 = -torch.mul(pred ** self.gamma, torch.mul(1 - target, torch.
            log(1 - pred + 1e-06)))
        temp2 = -torch.mul((1 - pred) ** self.gamma, torch.mul(target,
            torch.log(pred + 1e-06)))
        temp = temp1 + temp2
        CELoss = torch.sum(torch.mean(temp, (0, 1)))
        intersection_positive = torch.sum(pred * target, 1)
        cardinality_positive = torch.sum(torch.abs(pred) + torch.abs(target), 1
            )
        dice_positive = (intersection_positive + 1e-06) / (cardinality_positive
             + 1e-06)
        intersection_negative = torch.sum((1.0 - pred) * (1.0 - target), 1)
        cardinality_negative = torch.sum(2 - torch.abs(pred) - torch.abs(
            target), 1)
        dice_negative = (intersection_negative + 1e-06) / (cardinality_negative
             + 1e-06)
        temp3 = torch.mean(1.5 - dice_positive - dice_negative, 0)
        DICELoss = torch.sum(temp3)
        return CELoss + 1.0 * DICELoss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
