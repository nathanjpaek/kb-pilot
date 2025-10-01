import torch
import numpy as np
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, state_dim):
        super(CNN, self).__init__()
        self.state_dim = state_dim
        self.image_size = 64
        self.in_channels = 1
        self.kernel_size = 3
        self.padding = 0
        self.stride = 2
        self.pool_kernel = 2
        self.pool_stride = 2
        self.out_channels1 = 4
        self.out_channels2 = 8
        self.num_layers = 2
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels1, self.
            kernel_size, self.stride, self.padding)
        self.maxpool1 = nn.MaxPool2d(self.pool_kernel, self.pool_stride)
        self.conv2 = nn.Conv2d(self.out_channels1, self.out_channels2, self
            .kernel_size, self.stride, self.padding)
        self.maxpool2 = nn.MaxPool2d(self.pool_kernel, self.pool_stride)
        self.cnn_out_dim = self.calc_cnn_out_dim()
        self.linear = nn.Linear(self.cnn_out_dim, self.state_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

    def calc_cnn_out_dim(self):
        w = self.image_size
        h = self.image_size
        for l in range(self.num_layers):
            new_w = np.floor((w - self.kernel_size) / self.stride + 1)
            new_h = np.floor((h - self.kernel_size) / self.stride + 1)
            new_w = np.floor(new_w / self.pool_kernel)
            new_h = np.floor(new_h / self.pool_kernel)
            w = new_w
            h = new_h
        return int(w * h * 8)


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {'state_dim': 4}]
