import torch
from torch import nn


class OnnxToTorchModule:
    """
    Marker class for onnx2torch modules.
    """
    pass


class OnnxErf(nn.Module, OnnxToTorchModule):

    def forward(self, input_tensor: 'torch.Tensor') ->torch.Tensor:
        return torch.erf(input_tensor)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
