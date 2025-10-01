import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleAddMmModule(torch.nn.Module):

    def __init__(self, alpha=1, beta=1):
        super(SimpleAddMmModule, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, a, b, c):
        return (a + a).addmm(b, c)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
