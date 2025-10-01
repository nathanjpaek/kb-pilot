import torch
import torch.nn.functional as F
from torch import nn


class ClassificationNet(nn.Module):

    def __init__(self, num_classes=10, num_digits=2):
        super(ClassificationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024, num_classes * num_digits)

    def forward(self, out):
        out = self.conv1(out)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv4(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size()[0], -1)
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        out = torch.reshape(out, tuple(out.size()[:-1]) + (2, 10))
        None
        return out


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
