import torch
import torch.nn as nn


class evolution_area(nn.Module):
    """
    calcaulate the area of evolution curve
    """

    def __init__(self):
        super(evolution_area, self).__init__()

    def forward(self, mask_score, class_weight):
        curve_area = torch.sum(class_weight * mask_score)
        return curve_area


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
