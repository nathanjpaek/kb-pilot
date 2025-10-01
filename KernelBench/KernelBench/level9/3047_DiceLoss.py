import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, output, target):
        prediction = self.sigmoid(output)
        return 1 - 2 * torch.sum(prediction * target) / (torch.sum(
            prediction) + torch.sum(target) + 1e-07)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
