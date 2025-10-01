import torch
import torch.utils.data


class GrayScaleToRGB(torch.nn.Module):
    """
    Applies the transformation on an image to convert grayscale to rgb
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample):
        return sample.repeat(3, 1, 1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
