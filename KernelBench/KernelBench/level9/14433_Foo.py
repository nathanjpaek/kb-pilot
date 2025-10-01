import torch
import torch.jit
import torch.onnx
import torch.nn


class Foo(torch.nn.Module):

    def __init__(self):
        super(Foo, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(6, 16, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        y = self.conv2(x)
        return y


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
