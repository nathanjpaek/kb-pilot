import torch
from torch import nn
import torch.hub


def soft_dice_loss(outputs, targets, per_image=False, reduce=True, ohpm=
    False, ohpm_pixels=256 * 256):
    batch_size = outputs.size()[0]
    eps = 0.001
    if not per_image:
        batch_size = 1
    if ohpm:
        dice_target = targets.contiguous().view(-1).float()
        dice_output = outputs.contiguous().view(-1)
        loss_b = torch.abs(dice_target - dice_output)
        _, indc = loss_b.topk(ohpm_pixels)
        dice_target = dice_target[indc]
        dice_output = dice_output[indc]
        intersection = torch.sum(dice_output * dice_target)
        union = torch.sum(dice_output) + torch.sum(dice_target) + eps
        loss = 1 - (2 * intersection + eps) / union
        loss = loss.mean()
    else:
        dice_target = targets.contiguous().view(batch_size, -1).float()
        dice_output = outputs.contiguous().view(batch_size, -1)
        intersection = torch.sum(dice_output * dice_target, dim=1)
        union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1
            ) + eps
        loss = 1 - (2 * intersection + eps) / union
        if reduce:
            loss = loss.mean()
    return loss


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True, per_image=False,
        ohpm=False, ohpm_pixels=256 * 256):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.ohpm = ohpm
        self.ohpm_pixels = ohpm_pixels

    def forward(self, input, target):
        return soft_dice_loss(input, target, per_image=self.per_image, ohpm
            =self.ohpm, ohpm_pixels=self.ohpm_pixels)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
