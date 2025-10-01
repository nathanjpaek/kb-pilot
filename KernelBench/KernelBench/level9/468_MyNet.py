import torch
import torch.nn as nn


class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size
            =5, stride=2, padding=0)
        self.conv2_1 = nn.Conv2d(8, 16, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(16, 16, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(16, 24, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(24, 24, 3, 1, 0)
        self.conv4_1 = nn.Conv2d(24, 40, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(40, 80, 3, 1, 1)
        self.ip1 = nn.Linear(80 * 4 * 4, 128)
        self.ip2 = nn.Linear(128, 128)
        self.ip3 = nn.Linear(128, 42)
        self.prelu = nn.PReLU()
        self.avg_pool = nn.AvgPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        x = self.avg_pool(self.prelu(self.conv1_1(x)))
        x = self.prelu(self.conv2_1(x))
        x = self.prelu(self.conv2_2(x))
        x = self.avg_pool(x)
        x = self.prelu(self.conv3_1(x))
        x = self.prelu(self.conv3_2(x))
        x = self.avg_pool(x)
        x = self.prelu(self.conv4_1(x))
        x = self.prelu(self.conv4_2(x))
        ip = x.view(-1, 4 * 4 * 80)
        ip = self.prelu(self.ip1(ip))
        ip = self.prelu(self.ip2(ip))
        ip = self.ip3(ip)
        return ip


def get_inputs():
    return [torch.rand([4, 3, 144, 144])]


def get_init_inputs():
    return [[], {}]
