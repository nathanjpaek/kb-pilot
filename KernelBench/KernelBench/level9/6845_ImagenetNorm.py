import torch
import torch.nn as nn


class ImagenetNorm(nn.Module):

    def __init__(self, from_raw=True):
        """
        :param from_raw: whether the input image lies in the range of [0, 255]
        """
        super().__init__()
        self.from_raw = from_raw
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]),
            requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]),
            requires_grad=False)

    def forward(self, x: 'torch.Tensor'):
        if x.dtype != torch.float:
            x = x.float()
        x = x / 255 if self.from_raw else x
        mean = self.mean.view(1, 3, 1, 1)
        std = self.std.view(1, 3, 1, 1)
        return (x - mean) / std

    def extra_repr(self):
        return f'from_raw={self.from_raw}'


def get_inputs():
    return [torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {}]
