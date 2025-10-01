import torch
import torch.nn as nn
import torch.utils.data
import torch.random
import torch.nn.functional as F


class DiscountBlackLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, source, target):
        """

        Ignores pixels with all channels set to zero
        Probably Not suitable for greyscale images

        :param source: image from network
        :param target: ground truth
        :return: the loss
        """
        loss = F.mse_loss(source, target, reduction='none')
        mask = torch.sum(target, dim=1, keepdim=True)
        mask[mask > 0.0] = 1.0
        mask[mask == 0.0] = 1e-06
        loss = loss * mask
        return torch.mean(loss), loss, mask


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
