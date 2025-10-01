import torch
import torch.nn.functional as F


class Resize(torch.nn.Module):

    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.size = size
        self.mode = mode

    def forward(self, img):
        return F.interpolate(img, size=self.size, mode=self.mode,
            align_corners=False)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(size={self.size}, interpolation={self.mode})'
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
