import torch
import torch.nn as nn
import torch.nn.functional as F


class L1(nn.Module):

    def __init__(self):
        super(L1, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode='bilinear')
        loss = torch.mean(torch.abs(10.0 * real - 10.0 * fake))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
