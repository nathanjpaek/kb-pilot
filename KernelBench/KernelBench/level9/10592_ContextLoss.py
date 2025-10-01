import torch
import torch.nn.functional as F
import torch.nn as nn


class ContextLoss(nn.Module):

    def __init__(self):
        super(ContextLoss, self).__init__()

    def forward(self, generated, corrupted, weight_mask):
        c_loss = weight_mask * F.l1_loss(generated, corrupted)
        c_loss = c_loss.mean(dim=[0, 1, 2, 3])
        return c_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
