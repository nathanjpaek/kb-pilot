import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleMulModule(torch.nn.Module):

    def __init__(self):
        super(SimpleMulModule, self).__init__()

    def forward(self, left, right):
        other = left.mul(right.item() if right.size() == torch.Size([]) else
            right)
        return other.mul(other)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
