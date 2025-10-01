import torch
import torch.nn as nn
import torch.nn
import torch.utils.data
import torch.utils.tensorboard._pytorch_graph


class PassThroughOp(torch.nn.Module):
    """
    This is a pass-through op, used for purpose of making an op a no-op
    """

    def forward(self, inputx):
        return inputx


class SmallMnistNoDropoutWithPassThrough(nn.Module):

    def __init__(self):
        super(SmallMnistNoDropoutWithPassThrough, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pt1 = PassThroughOp()
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pt2 = PassThroughOp()
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.pt1(self.conv1(x)))
        x = self.conv2(x)
        x = self.relu2(self.pt2(x))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return self.log_softmax(x)


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
