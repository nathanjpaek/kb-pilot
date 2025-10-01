import torch
from torch import nn
import torch.nn.parallel
import torch.utils.data
import torch.utils


class Classifier(nn.Module):

    def __init__(self, num_classes, dim=128):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.fc1 = nn.Linear(self.dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, self.num_classes)
        self.nonlin = nn.LeakyReLU(0.2, inplace=False)
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.dim)
        x = self.fc1(x)
        x = self.nonlin(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.nonlin(x)
        x = self.drop(x)
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 128])]


def get_init_inputs():
    return [[], {'num_classes': 4}]
