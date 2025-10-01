import torch
import torch.nn.functional as F
import torch.fx


class QModReLU(torch.nn.Module):
    """
    Quaternion ModeReLU
    """

    def __init__(self, bias=0):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.Tensor([bias]))

    def forward(self, x):
        norm = x.norm()
        return F.relu(norm + self.bias) * (x / norm)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
