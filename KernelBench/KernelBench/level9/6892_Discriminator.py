import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 4, 2, 1, bias=False)
        self.act1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, 1, bias=False)
        self.act2 = nn.LeakyReLU(0.2, inplace=False)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.act3 = nn.LeakyReLU(0.2, inplace=False)
        self.conv4 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.act4 = nn.LeakyReLU(0.2, inplace=False)
        self.conv5 = nn.Conv2d(128, 128, 4, 2, 1, bias=False)
        self.act5 = nn.LeakyReLU(0.2, inplace=False)
        self.conv6 = nn.Conv2d(128, 128, 4, 2, 1, bias=False)
        self.act6 = nn.LeakyReLU(0.2, inplace=False)
        self.conv7 = nn.Conv2d(128, 2, 3, 1, bias=False)
        self.pool7 = nn.MaxPool2d(2, stride=2)

    def forward(self, labels):
        conv1 = self.act1(self.conv1(labels))
        conv2 = self.act2(self.conv2(conv1))
        conv3 = self.act3(self.conv3(conv2))
        conv4 = self.act4(self.conv4(conv3))
        conv5 = self.act5(self.conv5(conv4))
        conv6 = self.act6(self.conv6(conv5))
        conv7 = self.conv7(conv6)
        pool7 = self.pool7(conv7)
        return torch.sigmoid(pool7)


def get_inputs():
    return [torch.rand([4, 2, 256, 256])]


def get_init_inputs():
    return [[], {}]
