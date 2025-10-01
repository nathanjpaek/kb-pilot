import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistFeatureExtractor(nn.Module):

    def __init__(self, activation=F.leaky_relu):
        super(MnistFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.activation = activation

    def get_mtx(self):
        return None

    def forward(self, x):
        x = self.activation(F.max_pool2d(self.conv1(x), 2))
        x = self.activation(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        return x.view(x.size(0), -1)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
