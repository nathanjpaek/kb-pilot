import torch
import torch.nn.functional as F
import torch.nn as nn


class RMSE(nn.Module):

    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode='bilinear')
        loss = torch.sqrt(torch.mean(torch.abs(10.0 * real - 10.0 * fake) ** 2)
            )
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
