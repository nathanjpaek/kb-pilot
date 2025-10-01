import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleCat(nn.Module):
    """
    Unsample input and concat with contracting tensor
    """

    def __init__(self, ch):
        super(UpsampleCat, self).__init__()
        self.up_conv = nn.Conv2d(ch, ch // 2, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True)

    def forward(self, up, down):
        up = self.up_conv(up)
        up = self.up(up)
        up_w, up_h = up.size()[2:4]
        down_w, down_h = down.size()[2:4]
        dw = down_w + 4 - up_w
        dh = down_h + 4 - up_h
        down = F.pad(down, (2, 2, 2, 2))
        up = F.pad(up, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
        y = torch.cat([down, up], dim=1)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ch': 4}]
