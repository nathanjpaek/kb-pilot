import torch
import torchvision
import torch.utils.data
import torchvision.transforms
import torch.nn


class ConvertFloatToUint8(torch.nn.Module):
    """
    Converts a video from dtype float32 to dtype uint8.
    """

    def __init__(self):
        super().__init__()
        self.convert_func = torchvision.transforms.ConvertImageDtype(torch.
            uint8)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        assert x.dtype == torch.float or x.dtype == torch.half, 'image must have dtype torch.uint8'
        return self.convert_func(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
