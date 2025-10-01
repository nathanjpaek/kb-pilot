import torch
import torch.nn as nn
import torch.nn.functional as F


class net2(nn.Module):
    """
    """

    def __init__(self, n_classes=2):
        super(net2, self).__init__()
        if torch.cuda.is_available():
            torch.device('cuda')
        else:
            torch.device('cpu')
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(4, 64, 1)
        self.conv2 = nn.Conv2d(64, 256, 1)
        self.conv3 = nn.Conv2d(256, 128, 1)
        self.conv4 = nn.Conv2d(128, self.n_classes, 1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.logsoftmax(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
