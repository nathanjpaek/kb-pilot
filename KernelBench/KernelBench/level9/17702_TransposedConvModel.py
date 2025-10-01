import torch
import torch.nn
import torch.utils.data
import torch.utils.tensorboard._pytorch_graph
import torch.onnx.symbolic_caffe2


class TransposedConvModel(torch.nn.Module):

    def __init__(self):
        super(TransposedConvModel, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(10, 10, 3)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.ConvTranspose2d(10, 10, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


def get_inputs():
    return [torch.rand([4, 10, 4, 4])]


def get_init_inputs():
    return [[], {}]
