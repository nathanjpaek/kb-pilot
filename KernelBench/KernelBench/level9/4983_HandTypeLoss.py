import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F


class HandTypeLoss(nn.Module):

    def __init__(self):
        super(HandTypeLoss, self).__init__()

    def forward(self, hand_type_out, hand_type_gt, hand_type_valid):
        loss = F.binary_cross_entropy(hand_type_out, hand_type_gt,
            reduction='none')
        loss = loss.mean(1)
        loss = loss * hand_type_valid
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
