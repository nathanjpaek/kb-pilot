import torch
import torch.nn as nn


class QuadricLinearLoss(nn.Module):

    def __init__(self, clip_delta):
        super(QuadricLinearLoss, self).__init__()
        self.clip_delta = clip_delta

    def forward(self, y_pred, y_true, weights):
        td_error = y_true - y_pred
        td_error_abs = torch.abs(td_error)
        quadratic_part = torch.clamp(td_error_abs, max=self.clip_delta)
        linear_part = td_error_abs - quadratic_part
        loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        loss = torch.mean(loss * weights)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'clip_delta': 4}]
