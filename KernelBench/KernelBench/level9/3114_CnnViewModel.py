import torch
import torch.nn as nn


class CnnViewModel(nn.Module):

    def __init__(self, out_dim=10):
        super(CnnViewModel, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
            stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=
            5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 7 * 7, out_dim)

    def forward(self, X):
        X = X.view(-1, 1, 28, 28)
        out = self.cnn1(X)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


def get_inputs():
    return [torch.rand([4, 1, 28, 28])]


def get_init_inputs():
    return [[], {}]
