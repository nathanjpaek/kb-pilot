import torch
import torch.nn
import torch.onnx


class MegatronGelu(torch.nn.Module):

    def forward(self, x):
        return x * 0.5 * (torch.erf(x / 1.41421) + 1.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
