import torch
import torch.nn as nn
import torch.cuda


class NodeMaxpool3by3(nn.Module):

    def __init__(self):
        super(NodeMaxpool3by3, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def init_weights(self):
        pass

    def forward(self, x):
        return self.maxpool(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
