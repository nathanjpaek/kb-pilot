import torch
import torch.onnx
import torch.nn


class SimpleStackModel(torch.nn.Module):

    def __init__(self):
        super(SimpleStackModel, self).__init__()

    def forward(self, a, b):
        c = torch.stack((a, b), 0)
        d = torch.stack((c, c), 1)
        return torch.stack((d, d), 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
