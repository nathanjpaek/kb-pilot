import torch
from torch import nn


class OnnxToTorchModule:
    """
    Marker class for onnx2torch modules.
    """
    pass


class OnnxSoftmaxV1V11(nn.Module, OnnxToTorchModule):

    def __init__(self, axis: 'int'=1, is_log: 'bool'=False):
        super().__init__()
        self.axis = axis
        self.is_log = is_log

    def forward(self, input_tensor: 'torch.Tensor') ->torch.Tensor:
        shape = input_tensor.shape
        result = torch.flatten(input_tensor, start_dim=self.axis)
        result = torch.log_softmax(result, -1
            ) if self.is_log else torch.softmax(result, -1)
        return torch.reshape(result, shape)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
