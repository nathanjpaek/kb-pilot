import torch
import torch.cuda
import torch.nn
import torch.utils.data
import torch.fx
import torch.utils.tensorboard._pytorch_graph
import torch.onnx.symbolic_caffe2


class SoftMaxAvgPoolModel(torch.nn.Module):

    def __init__(self):
        super(SoftMaxAvgPoolModel, self).__init__()
        self.sfmax = torch.nn.Softmax(dim=1)
        self.avgpool = torch.nn.AvgPool2d(3)

    def forward(self, inp):
        x = self.sfmax(inp)
        return self.avgpool(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
