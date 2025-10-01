import torch
import torch.nn as nn


class SuperpointBackbone(nn.Module):
    """ SuperPoint backbone. """

    def __init__(self):
        super(SuperpointBackbone, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4 = 64, 64, 128, 128
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1
            )
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1,
            padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1,
            padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1,
            padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1,
            padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1,
            padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1,
            padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1,
            padding=1)

    def forward(self, input_images):
        x = self.relu(self.conv1a(input_images))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
