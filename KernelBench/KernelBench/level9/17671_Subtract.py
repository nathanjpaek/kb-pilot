import torch
import torch.nn
import torch.utils.data
import torch.utils.tensorboard._pytorch_graph
import torch.onnx.symbolic_caffe2


class Subtract(torch.nn.Module):
    """ Subtract module for a functional subtract"""

    def forward(self, x, y):
        """
        Forward-pass routine for subtact op
        """
        return x - y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
