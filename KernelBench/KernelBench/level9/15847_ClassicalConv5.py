import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.utils.prune
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data


class ClassicalConv5(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.avg_pool2d(x, 4)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.mean(-1).squeeze()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
