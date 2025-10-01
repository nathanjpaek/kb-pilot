import torch
import torch.jit
import torch.nn.functional as F
import torch.onnx
import torch.nn


class SimpleSoftPlusModel(torch.nn.Module):

    def __init__(self):
        super(SimpleSoftPlusModel, self).__init__()

    def forward(self, tensor):
        tensor = tensor + tensor
        return F.softplus(tensor)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
