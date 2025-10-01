import torch
import torch.nn as nn


class FeatNet(nn.Module):

    def __init__(self):
        super(FeatNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=
            (3, 7), stride=1, padding=(1, 3), bias=False)
        self.tanh1 = nn.Tanh()
        self.Pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size
            =(3, 5), stride=1, padding=(1, 2), bias=False)
        self.tanh2 = nn.Tanh()
        self.Upsample2 = nn.ConvTranspose2d(in_channels=32, out_channels=32,
            kernel_size=4, stride=2, padding=1, groups=32)
        self.Pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size
            =(3, 3), stride=1, padding=(1, 1), bias=False)
        self.tanh3 = nn.Tanh()
        self.Upsample3 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
            kernel_size=8, stride=4, padding=2, bias=False, groups=64)
        self.conv4 = nn.Conv2d(in_channels=112, out_channels=1, kernel_size
            =3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh1(x)
        stack1 = x
        x = self.Pool1(x)
        x = self.conv2(x)
        x = self.tanh2(x)
        stack2 = self.Upsample2(x)
        x = self.Pool2(x)
        x = self.conv3(x)
        x = self.tanh3(x)
        stack3 = self.Upsample3(x)
        x = torch.concat((stack1, stack2, stack3), 1)
        output = self.conv4(x)
        return output


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
