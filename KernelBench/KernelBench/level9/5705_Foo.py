import torch
import torch.nn.parallel
import torch.utils.data
import torch.onnx
import torch.fx
import torch.optim
import torch.utils.data.distributed


def add_lowp(a: 'torch.Tensor', b: 'torch.Tensor'):
    a, b = a.float(), b.float()
    c = a + b
    return c.half()


def sigmoid_lowp(x: 'torch.Tensor'):
    x = x.float()
    x = x.sigmoid()
    return x.half()


class Foo(torch.nn.Module):

    def forward(self, x, y):
        x = sigmoid_lowp(x)
        y = sigmoid_lowp(y)
        return add_lowp(x, y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
