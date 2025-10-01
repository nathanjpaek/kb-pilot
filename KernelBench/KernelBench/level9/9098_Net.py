import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_basic(nn.Module):
    """基础网络，仅包含保存、加载模型的功能"""

    def __init__(self):
        super(Net_basic, self).__init__()

    def load(self, path):
        """加载指定模型"""
        self.load_state_dict(torch.load(path))

    def save(self, path):
        """保存模型"""
        torch.save(self.state_dict(), path)


class Net(Net_basic):

    def __init__(self, inp, out, pooling=[1, 1, 1, 1]):
        super(Net, self).__init__()
        self.pooling = pooling
        self.conv1 = nn.Conv2d(inp, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv9 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv10 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv11 = nn.Conv2d(64, out, 1, padding=0)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        if self.pooling[0] == 1:
            x1 = F.max_pool2d(x1, 2)
        x2 = F.relu(self.conv2(x1))
        if self.pooling[1] == 1:
            x2 = F.max_pool2d(x2, 2)
        x3 = F.relu(self.conv3(x2))
        if self.pooling[2] == 1:
            x3 = F.max_pool2d(x3, 2)
        x4 = F.relu(self.conv4(x3))
        if self.pooling[3] == 1:
            x4 = F.max_pool2d(x4, 2)
        x5 = F.relu(self.conv5(x4))
        if self.pooling[3] == 1:
            x5 = self.up(x5)
        x6 = F.relu(self.conv6(x5))
        if self.pooling[2] == 1:
            x6 = self.up(x6)
        x7 = F.relu(self.conv7(x6))
        if self.pooling[1] == 1:
            x7 = self.up(x7)
        x8 = F.relu(self.conv8(x7))
        if self.pooling[0] == 1:
            x8 = self.up(x8)
        x9 = F.relu(self.conv9(x8))
        x10 = F.relu(self.conv10(x9))
        x11 = self.conv11(x10)
        y = torch.sigmoid(x11)
        return y


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'inp': 4, 'out': 4}]
