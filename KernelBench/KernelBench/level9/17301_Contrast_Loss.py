import torch
import torch.nn as nn


class Contrast_Loss(nn.Module):

    def __init__(self, margin=0.5):
        super(Contrast_Loss, self).__init__()
        self.margin = margin

    def forward(self, inputs, target):
        R = (target.unsqueeze(0) == target.unsqueeze(1)).float()
        distance = ((inputs.unsqueeze(0) - inputs.unsqueeze(1)) ** 2).sum(dim=2
            )
        loss = R * distance + (1.0 - R) * (self.margin - distance).clamp(min
            =0.0)
        loss_mean = loss.sum() / (inputs.size(0) * (inputs.size(0) - 1.0) * 2)
        return loss_mean


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
