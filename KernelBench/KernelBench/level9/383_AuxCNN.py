import torch
from torch import nn
from torch.nn import functional as F


class AuxCNN(nn.Module):
    """
    basic structure similar to the CNN
    input is splited into two 1*14*14 images for separating training, use different parameters
    softmax for the auxiliary output layers
    """

    def __init__(self):
        super(AuxCNN, self).__init__()
        self.conv11 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv21 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc11 = nn.Linear(256, 200)
        self.fc21 = nn.Linear(200, 10)
        self.conv12 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv22 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc12 = nn.Linear(256, 200)
        self.fc22 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(20, 2)

    def convs(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        return x

    def forward(self, x1, x2):
        x1 = F.relu(F.max_pool2d(self.conv11(x1), kernel_size=2))
        x1 = F.relu(F.max_pool2d(self.conv21(x1), kernel_size=2))
        x1 = x1.view(-1, 256)
        x1 = F.relu(self.fc11(x1))
        x1 = self.fc21(x1)
        x2 = F.relu(F.max_pool2d(self.conv12(x2), kernel_size=2))
        x2 = F.relu(F.max_pool2d(self.conv22(x2), kernel_size=2))
        x2 = x2.view(-1, 256)
        x2 = F.relu(self.fc12(x2))
        x2 = self.fc22(x2)
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(x)
        x = torch.sigmoid(self.fc3(x))
        return x, x1, x2


def get_inputs():
    return [torch.rand([4, 1, 64, 64]), torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
