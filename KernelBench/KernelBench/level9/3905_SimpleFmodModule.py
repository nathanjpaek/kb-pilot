import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleFmodModule(torch.nn.Module):

    def __init__(self):
        super(SimpleFmodModule, self).__init__()

    def forward(self, a, b):
        if b.size() == torch.Size([]):
            c = a.fmod(b.item())
        else:
            c = a.fmod(b)
        return c.fmod(torch.tensor(1.0, dtype=c.dtype))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
