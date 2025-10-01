import torch
import torch.nn.functional as F
import torch.nn as nn


class Evidential_layer(nn.Module):

    def __init__(self, in_dim, num_classes):
        super(Evidential_layer, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(in_dim, 2 * self.num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        return self.relu(x)


class MNISTmodel(nn.Module):

    def __init__(self, num_classes, edl, dropout=True):
        super(MNISTmodel, self).__init__()
        self.use_dropout = dropout
        k, m = 8, 80
        km = (64 - 2 * (k - 1)) ** 2 * m
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 20, kernel_size=k)
        self.conv2 = nn.Conv2d(20, m, kernel_size=k)
        self.fc1 = nn.Linear(km, 500)
        if edl:
            self.fc2 = Evidential_layer(500, self.num_classes)
        else:
            self.fc2 = nn.Linear(500, self.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 1))
        x = F.relu(F.max_pool2d(self.conv2(x), 1))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {'num_classes': 4, 'edl': 4}]
