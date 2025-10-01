import torch
import torch.nn as nn
import torch.nn.functional as F


class vgg11_modified(nn.Module):

    def __init__(self, num_classes=20):
        super(vgg11_modified, self).__init__()
        self.num_classes = num_classes
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.pool = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        self.conv1_1 = nn.Conv2d(3, 64, (3, 3))
        self.conv2_1 = nn.Conv2d(64, 128, (3, 3))
        self.conv3_1 = nn.Conv2d(128, 256, (3, 3))
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3))
        self.conv4_1 = nn.Conv2d(256, 512, (3, 3))

    def forward(self, vector):
        vector = self.pad(self.pool(F.relu(self.conv1_1(vector))))
        vector = self.pad(self.pool(F.relu(self.conv2_1(vector))))
        vector = self.pad(F.relu(self.conv3_1(vector)))
        vector = self.pad(self.pool(F.relu(self.conv3_2(vector))))
        vector = self.pad(F.relu(self.conv4_1(vector)))
        vector = torch.flatten(vector, 1)
        return vector


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
