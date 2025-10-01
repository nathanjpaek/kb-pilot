import torch
import numpy as np
from torch import nn


class ycbcr_to_rgb_jpeg(nn.Module):
    """ Converts YCbCr image to RGB JPEG
    Input:
        image(tensor): batch x height x width x 3
    Outpput:
        result(tensor): batch x 3 x height x width
    """

    def __init__(self):
        super(ycbcr_to_rgb_jpeg, self).__init__()
        matrix = np.array([[1.0, 0.0, 1.402], [1, -0.344136, -0.714136], [1,
            1.772, 0]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0, -128.0, -128.0]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        result.view(image.shape)
        return result.permute(0, 3, 1, 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 3])]


def get_init_inputs():
    return [[], {}]
