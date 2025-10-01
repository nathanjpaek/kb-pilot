import torch
import torch.nn as nn
import torch.hub


def dice_loss(input, target):
    smooth = 1.0
    input = torch.sigmoid(input)
    if input.dim() == 4:
        B, C, _H, _W = input.size()
        iflat = input.view(B * C, -1)
        tflat = target.view(B * C, -1)
    else:
        assert input.dim() == 3
        B, _H, _W = input.size()
        iflat = input.view(B, -1)
        tflat = target.view(B, -1)
    intersection = (iflat * tflat).sum(dim=1)
    loss = 1 - (2.0 * intersection + smooth) / (iflat.sum(dim=1) + tflat.
        sum(dim=1) + smooth)
    loss = loss.mean()
    return loss


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def forward(self, input, target):
        return dice_loss(input, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
