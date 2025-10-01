import torch
import torch.nn.parallel
import torch.utils.data
import torch.onnx
import torch.fx
import torch.optim
import torch.utils.data.distributed


class M(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        y = torch.cat([x, y])
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
