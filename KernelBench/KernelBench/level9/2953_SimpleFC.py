import torch
import torch.nn as nn
import torch.onnx


class SimpleFC(nn.Module):

    def __init__(self, input_size, num_classes, name='SimpleFC'):
        super(SimpleFC, self).__init__()
        self.FC = nn.Parameter(torch.randn([input_size, num_classes]))
        self.FCbias = nn.Parameter(torch.randn([num_classes]))

    def forward(self, input):
        return torch.matmul(input, self.FC) + self.FCbias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'num_classes': 4}]
