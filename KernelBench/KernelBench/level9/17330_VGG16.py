import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))
        self.conv2_1 = nn.Conv2d(64, 128, 3)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1))
        self.conv3_1 = nn.Conv2d(128, 256, 3)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1))
        self.conv4_1 = nn.Conv2d(256, 512, 3)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1))
        self.conv5_1 = nn.Conv2d(512, 512, 3)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))

    def forward(self, x):
        x.size(0)
        out = self.conv1_1(x)
        out = F.relu(out)
        out = self.conv1_2(out)
        out = F.relu(out)
        out1 = self.maxpool1(out)
        out = self.conv2_1(out1)
        out = F.relu(out)
        out = self.conv2_2(out)
        out = F.relu(out)
        out2 = self.maxpool2(out)
        out = self.conv3_1(out2)
        out = F.relu(out)
        out = self.conv3_2(out)
        out = F.relu(out)
        out = self.conv3_3(out)
        out = F.relu(out)
        out3 = self.maxpool3(out)
        out = self.conv4_1(out3)
        out = F.relu(out)
        out = self.conv4_2(out)
        out = F.relu(out)
        out = self.conv4_3(out)
        out = F.relu(out)
        out4 = self.maxpool4(out)
        out = self.conv5_1(out4)
        out = F.relu(out)
        out = self.conv5_2(out)
        out = F.relu(out)
        out = self.conv5_3(out)
        out = F.relu(out)
        out5 = self.maxpool5(out)
        return out5


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
