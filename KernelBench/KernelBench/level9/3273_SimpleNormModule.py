import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleNormModule(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(SimpleNormModule, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, tensor):
        return torch.norm(tensor, *self.args, **self.kwargs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
