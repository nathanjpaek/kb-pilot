import torch
import torch.nn as nn
import torch.nn.functional as F


class NetTan2018(nn.Module):

    def __init__(self, in_channels=3, out_classes=2):
        super(NetTan2018, self).__init__()
        oc = 16
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=oc,
            kernel_size=(3, 3), padding=0)
        self.max1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=oc, out_channels=oc * 2,
            kernel_size=(3, 3), padding=0)
        self.max2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=oc * 2, out_channels=oc * 2,
            kernel_size=(3, 3), padding=0)
        self.max3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=oc * 2, out_channels=oc * 4,
            kernel_size=(3, 3), padding=0)
        self.conv5 = nn.Conv2d(in_channels=oc * 4, out_channels=oc * 4,
            kernel_size=(3, 3), padding=0)
        self.max5 = nn.MaxPool2d(2, 2)
        self.conv6 = nn.Conv2d(in_channels=oc * 4, out_channels=oc * 8,
            kernel_size=(3, 3), padding=0)
        self.conv7 = nn.Conv2d(in_channels=oc * 8, out_channels=oc * 8,
            kernel_size=(3, 3), padding=0)
        self.hidden1 = nn.Linear(in_features=4 * 4 * 128, out_features=128)
        self.hidden2 = nn.Linear(in_features=128, out_features=64)
        self.final = nn.Linear(in_features=64, out_features=out_classes)

    def forward(self, x):
        x = self.max1(F.relu(self.conv1(x)))
        x = self.max2(F.relu(self.conv2(x)))
        x = self.max3(F.relu(self.conv3(x)))
        x = self.max5(F.relu(self.conv5(F.relu(self.conv4(x)))))
        x = F.relu(self.conv7(F.relu(self.conv6(x))))
        x = x.view(-1, 4 * 4 * 128)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.final(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 144, 144])]


def get_init_inputs():
    return [[], {}]
