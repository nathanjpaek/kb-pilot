import torch
import torch.nn as nn
import torch.utils.data


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.Conv1 = nn.Conv2d(1, 15, 9, 1, 0)
        self.Relu1 = nn.ReLU()
        self.MaxPool1 = nn.MaxPool2d(2)
        self.Conv2 = nn.Conv2d(15, 20, 5, 1, 0)
        self.Relu2 = nn.ReLU()
        self.MaxPool2 = nn.MaxPool2d(2)
        self.Fc1 = nn.Linear(180, 100)
        self.Relu3 = nn.ReLU()
        self.Fc2 = nn.Linear(100, 10)

    def forward(self, data):
        x = self.Conv1(data)
        x = self.Relu1(x)
        x = self.MaxPool1(x)
        x = self.Conv2(x)
        x = self.Relu2(x)
        x = self.MaxPool2(x)
        x = x.view(-1, 180)
        x = self.Fc1(x)
        x = self.Relu3(x)
        x = self.Fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
