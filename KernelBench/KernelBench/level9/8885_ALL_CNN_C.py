import torch
from torch import nn
import torch.nn.functional as F


class ALL_CNN_C(nn.Module):

    def __init__(self, num_classes=10):
        super(ALL_CNN_C, self).__init__()
        self.model_name = 'ALL_CNN_C'
        self.dp0 = nn.Dropout2d(p=0.2)
        self.conv1 = nn.Conv2d(3, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, stride=2, padding=1)
        self.dp1 = nn.Dropout2d(p=0.5)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
        self.dp2 = nn.Dropout2d(p=0.5)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=0)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv9 = nn.Conv2d(192, 10, 1)
        self.avg = nn.AvgPool2d(6)

    def forward(self, x):
        x = self.dp0(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dp1(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = F.relu(x)
        x = self.avg(x)
        x = torch.squeeze(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
