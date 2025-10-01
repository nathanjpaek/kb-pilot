import torch
from torch import nn


class OnnxToTorchModule:
    """
    Marker class for onnx2torch modules.
    """
    pass


class OnnxHardSigmoid(nn.Module, OnnxToTorchModule):

    def __init__(self, alpha: 'float'=0.2, beta: 'float'=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input_tensor: 'torch.Tensor') ->torch.Tensor:
        return torch.clip(self.alpha * input_tensor + self.beta, min=0.0,
            max=1.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
