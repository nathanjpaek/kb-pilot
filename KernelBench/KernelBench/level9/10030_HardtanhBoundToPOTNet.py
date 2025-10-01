import torch
from torch.nn import Conv2d
from torch.nn import Hardtanh
from torch.nn.functional import relu
from torch.nn.functional import hardtanh
import torch.nn.functional


class HardtanhBoundToPOTNet(torch.nn.Module):

    def __init__(self):
        super(HardtanhBoundToPOTNet, self).__init__()
        self.conv1 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.hardtanh1 = Hardtanh(min_val=0.0, max_val=6.0)
        self.conv2 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv3 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.hardtanh2 = Hardtanh(min_val=-2.0, max_val=6.0)
        self.conv4 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv5 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv6 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv7 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.hardtanh3 = Hardtanh(min_val=0.0, max_val=4.0)
        self.conv8 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv9 = Conv2d(3, 3, kernel_size=1, stride=1)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.hardtanh1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.hardtanh2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = hardtanh(x, min_val=0.0, max_val=6.0)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.hardtanh3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
