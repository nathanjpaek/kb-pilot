import torch
import torch.jit
import torch.onnx
import torch.nn


class RepeatModule(torch.nn.Module):

    def __init__(self, repeats):
        super(RepeatModule, self).__init__()
        self.repeats = repeats

    def forward(self, tensor):
        tensor = tensor + tensor
        return tensor.repeat(self.repeats)


def get_inputs():
    return [torch.rand([4])]


def get_init_inputs():
    return [[], {'repeats': 4}]
