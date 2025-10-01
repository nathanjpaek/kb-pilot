import torch
import numpy as np
import torch.nn as nn


class ClassificationSubNet(nn.Module):

    def __init__(self, in_channels, num_classes, num_anchors=9):
        super().__init__()
        self.num_classes = num_classes
        self.conv2d_1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        nn.init.normal_(self.conv2d_1.weight.data, std=0.01)
        nn.init.zeros_(self.conv2d_1.bias.data)
        self.conv2d_2 = nn.Conv2d(256, 256, 3, padding=1)
        nn.init.normal_(self.conv2d_2.weight.data, std=0.01)
        nn.init.zeros_(self.conv2d_2.bias.data)
        self.conv2d_3 = nn.Conv2d(256, 256, 3, padding=1)
        nn.init.normal_(self.conv2d_3.weight.data, std=0.01)
        nn.init.zeros_(self.conv2d_3.bias.data)
        self.conv2d_4 = nn.Conv2d(256, 256, 3, padding=1)
        nn.init.normal_(self.conv2d_4.weight.data, std=0.01)
        nn.init.zeros_(self.conv2d_4.bias.data)
        self.conv2d_5 = nn.Conv2d(256, num_classes * num_anchors, 3, padding=1)
        nn.init.normal_(self.conv2d_5.weight.data, std=0.01)
        nn.init.constant_(self.conv2d_5.bias.data, val=np.log(1 / 99))

    def forward(self, x):
        x = self.conv2d_1(x)
        x = nn.functional.relu(x)
        x = self.conv2d_2(x)
        x = nn.functional.relu(x)
        x = self.conv2d_3(x)
        x = nn.functional.relu(x)
        x = self.conv2d_4(x)
        x = nn.functional.relu(x)
        x = self.conv2d_5(x)
        x = torch.sigmoid(x)
        x = x.permute(0, 2, 3, 1)
        return x.reshape(x.size(0), -1, self.num_classes)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'num_classes': 4}]
