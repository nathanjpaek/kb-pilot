import torch
import torch.nn as nn


class IndexedSegmentationMap(nn.Module):
    """
    Takes the raw logits from the n-channel output convolution and uses argmax to convert to an indexed output map.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return torch.argmax(x.squeeze(), dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
