import torch
import torch.nn as nn


class length_evolution(nn.Module):
    """
    calcaulate the length of evolution curve by the gradient
    """

    def __init__(self, func='l1'):
        super(length_evolution, self).__init__()
        self.func = func

    def forward(self, mask_score, class_weight):
        gradient_H = torch.abs(mask_score[:, :, 1:, :] - mask_score[:, :, :
            -1, :])
        gradient_W = torch.abs(mask_score[:, :, :, 1:] - mask_score[:, :, :,
            :-1])
        if self.func == 'l2':
            gradient_H = gradient_H * gradient_H
            gradient_W = gradient_W * gradient_W
        curve_length = torch.sum(class_weight * gradient_H) + torch.sum(
            class_weight * gradient_W)
        return curve_length


def get_inputs():
    return [torch.rand([4, 4, 2, 2]), torch.rand([4, 4, 2, 2])]


def get_init_inputs():
    return [[], {}]
