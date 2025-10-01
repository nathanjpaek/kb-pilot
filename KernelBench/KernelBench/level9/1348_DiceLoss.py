import torch
from torch.nn.modules.loss import _Loss


class DiceLoss(_Loss):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.axis = 2, 3, 4
        self.smooth = 1e-07

    def forward(self, input, target):
        return 1.0 - self.dice_score(input, target)

    def dice_score(self, input, target):
        numerator = torch.sum(input * target, dim=self.axis)
        dice = 2.0 * (numerator + self.smooth) / (torch.sum(input, dim=self
            .axis) + torch.sum(target, dim=self.axis) + self.smooth)
        return torch.mean(dice)


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
