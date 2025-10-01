import torch
import torch.nn as nn
import torch.nn.functional as F


class PSNR(nn.Module):

    def __init__(self, max_val=1.0, mode='Y'):
        super(PSNR, self).__init__()
        self.max_val = max_val
        self.mode = mode

    def forward(self, x, y):
        if self.mode == 'Y' and x.shape[1] == 3 and y.shape[1] == 3:
            x = kornia.color.rgb_to_grayscale(x)
            y = kornia.color.rgb_to_grayscale(y)
        mse = F.mse_loss(x, y, reduction='mean')
        psnr = 10 * torch.log10(self.max_val ** 2 / mse)
        return psnr


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
