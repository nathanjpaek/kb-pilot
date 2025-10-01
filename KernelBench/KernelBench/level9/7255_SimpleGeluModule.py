import torch
import torch.nn.functional as F
import torch.jit
import torch.onnx
import torch.nn


class SimpleGeluModule(torch.nn.Module):

    def forward(self, tensor):
        return F.gelu(tensor + tensor)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
