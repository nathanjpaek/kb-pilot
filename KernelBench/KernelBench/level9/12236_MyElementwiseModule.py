import torch
import torch.nn.parallel
import torch.utils.data
import torch.onnx
import torch.fx
import torch.optim
import torch.utils.data.distributed


class MyElementwiseModule(torch.nn.Module):

    def forward(self, x, y):
        return x * y + y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
