import torch
from torch import nn


class PixelDynamicsLoss(nn.Module):

    def __init__(self, diff_pp=False):
        super().__init__()
        self.diff_pp = diff_pp

    def forward(self, target_t, target_tk, pred_t, pred_tk):
        if self.diff_pp:
            loss = ((target_t - target_tk).abs() - (pred_t.detach() -
                pred_tk).abs()).mean() ** 2
        else:
            loss = ((target_t - target_tk).abs().mean() - (pred_t.detach() -
                pred_tk).abs().mean()) ** 2
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
