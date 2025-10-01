import torch
import torch.nn as nn


class ResizeCat(nn.Module):

    def __init__(self, **kwargs):
        super(ResizeCat, self).__init__()

    def forward(self, at1, at3, at5):
        _N, _C, H, W = at1.size()
        resized_at3 = nn.functional.interpolate(at3, (H, W))
        resized_at5 = nn.functional.interpolate(at5, (H, W))
        cat_at = torch.cat((at1, resized_at3, resized_at5), dim=1)
        return cat_at


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
