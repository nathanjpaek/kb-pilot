from _paritybench_helpers import _mock_config
import torch
from torch import nn
import torch.nn.functional as F


class CNNCifar100(nn.Module):

    def __init__(self, args):
        super(CNNCifar100, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.6)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, args.num_classes)
        self.cls = args.num_classes
        self.weight_keys = [['fc1.weight', 'fc1.bias'], ['fc2.weight',
            'fc2.bias'], ['fc3.weight', 'fc3.bias'], ['conv2.weight',
            'conv2.bias'], ['conv1.weight', 'conv1.bias']]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 3, 32, 32])]


def get_init_inputs():
    return [[], {'args': _mock_config(num_classes=4)}]
