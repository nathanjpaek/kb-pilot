import torch
import torch.utils.data


def smooth_l1_loss(pred, target, weight, beta):
    val = target - pred
    abs_val = val.abs()
    smooth_mask = abs_val < beta
    return weight * torch.where(smooth_mask, 0.5 / beta * val ** 2, abs_val -
        0.5 * beta).sum(dim=-1)


class SmoothL1Loss(torch.nn.Module):

    def __init__(self, beta=1.0 / 9):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, input, target, size_average=True):
        return smooth_l1_loss(input, target, self.beta, size_average)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
