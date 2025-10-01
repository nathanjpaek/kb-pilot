import torch
from torch import nn


class UNET(nn.Module):

    def __init__(self):
        super(UNET, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16,
            kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=32,
            kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=32,
            kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=64,
            kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=64,
            kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=128,
            kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=128, out_channels=128,
            kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(in_channels=128, out_channels=256,
            kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.upsample1 = nn.ConvTranspose2d(in_channels=256, out_channels=
            128, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(in_channels=256, out_channels=128,
            kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(in_channels=128, out_channels=128,
            kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()
        self.upsample2 = nn.ConvTranspose2d(in_channels=128, out_channels=
            64, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(in_channels=128, out_channels=64,
            kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(in_channels=64, out_channels=64,
            kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()
        self.upsample3 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
            kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(in_channels=64, out_channels=32,
            kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(in_channels=32, out_channels=32,
            kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU()
        self.upsample4 = nn.ConvTranspose2d(in_channels=32, out_channels=16,
            kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(in_channels=32, out_channels=16,
            kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1)
        self.relu10 = nn.ReLU()

    def forward(self, x):
        c1 = self.relu1(self.conv1_2(self.conv1_1(x)))
        p1 = self.pool1(c1)
        c2 = self.relu2(self.conv2_2(self.conv2_1(p1)))
        p2 = self.pool2(c2)
        c3 = self.relu3(self.conv3_2(self.conv3_1(p2)))
        p3 = self.pool3(c3)
        c4 = self.relu4(self.conv4_2(self.conv4_1(p3)))
        p4 = self.pool4(c4)
        c5 = self.relu5(self.conv5_2(self.conv5_1(p4)))
        u6 = self.upsample1(c5)
        u6 = torch.cat((u6, c4), dim=1)
        c6 = self.relu6(self.conv6_2(self.conv6_1(u6)))
        u7 = self.upsample2(c6)
        u7 = torch.cat((u7, c3), dim=1)
        c7 = self.relu7(self.conv7_2(self.conv7_1(u7)))
        u8 = self.upsample3(c7)
        u8 = torch.cat((u8, c2), dim=1)
        c8 = self.relu8(self.conv8_2(self.conv8_1(u8)))
        u9 = self.upsample4(c8)
        u9 = torch.cat((u9, c1), dim=1)
        c9 = self.relu9(self.conv9_2(self.conv9_1(u9)))
        c10 = self.relu10(self.conv10(c9))
        return c10


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
