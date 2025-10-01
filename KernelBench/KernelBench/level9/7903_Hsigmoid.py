import torch
import torch.nn as nn
import torch.quantization


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.float_op = nn.quantized.FloatFunctional()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        relu6 = self.relu6(self.float_op.add_scalar(x, 3.0))
        return self.float_op.mul_scalar(relu6, 1 / 6.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
