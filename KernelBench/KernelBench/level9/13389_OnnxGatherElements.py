import torch
from torch import nn


class OnnxToTorchModule:
    """
    Marker class for onnx2torch modules.
    """
    pass


class OnnxGatherElements(nn.Module, OnnxToTorchModule):

    def __init__(self, axis: 'int'=0):
        super().__init__()
        self.axis = axis

    def forward(self, input_tensor: 'torch.Tensor', indices: 'torch.Tensor'
        ) ->torch.Tensor:
        return torch.gather(input_tensor, dim=self.axis, index=indices)


def get_inputs():
    return [torch.ones([4], dtype=torch.int64), torch.ones([4], dtype=torch
        .int64)]


def get_init_inputs():
    return [[], {}]
