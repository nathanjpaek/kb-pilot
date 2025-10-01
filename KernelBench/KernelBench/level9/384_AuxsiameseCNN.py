import torch
from torch import nn
from torch.nn import functional as F


class AuxsiameseCNN(nn.Module):
    """
    basic structure similar to the CNN
    input is splited into two 1*14*14 images for separating training, share the same parameters
    softmax for the auxiliary output layers
    """

    def __init__(self):
        super(AuxsiameseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(20, 2)

    def convs(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        return x

    def forward(self, x1, x2):
        x1 = self.convs(x1)
        x1 = x1.view(-1, 256)
        x1 = F.relu(self.fc1(x1))
        x1 = self.fc2(x1)
        x2 = self.convs(x2)
        x2 = x2.view(-1, 256)
        x2 = F.relu(self.fc1(x2))
        x2 = self.fc2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(x)
        x = torch.sigmoid(self.fc3(x))
        return x, x1, x2


def get_inputs():
    return [torch.rand([4, 1, 64, 64]), torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
