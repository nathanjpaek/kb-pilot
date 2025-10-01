import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Model(nn.Module):

    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 33)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = x.view(-1, 4096)
        x = self.fc1(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
