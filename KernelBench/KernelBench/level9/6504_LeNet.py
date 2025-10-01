import torch
import torch.nn as nn


class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=2)
        self.conv_2 = nn.Conv2d(6, 16, kernel_size=5, stride=2)
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, 7)

    def get_resnet_layer(self, block, n_blocks, channels, stride=1):
        layers = []
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        layers.append(block(self.in_channels, channels, stride, downsample))
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))
        self.in_channels = block.expansion * channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.sigmoid(x)
        x = self.avgpool(x)
        x = self.conv_2(x)
        x = self.sigmoid(x)
        x = self.avgpool(x)
        x = nn.Flatten()(x)
        x = self.fc_1(x)
        x = self.sigmoid(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)
        x = self.fc_3(x)
        h = x.view(x.shape[0], -1)
        return x, h


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
