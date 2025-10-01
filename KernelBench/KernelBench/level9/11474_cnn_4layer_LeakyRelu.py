import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn_4layer_LeakyRelu(nn.Module):

    def __init__(self, in_ch, in_dim, width=2, linear_size=256, alpha=0.1):
        super(cnn_4layer_LeakyRelu, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 4 * width, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4 * width, 8 * width, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(8 * width * (in_dim // 4) * (in_dim // 4),
            linear_size)
        self.fc2 = nn.Linear(linear_size, 10)
        self.alpha = alpha

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), self.alpha)
        x = F.leaky_relu(self.conv2(x), self.alpha)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), self.alpha)
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'in_dim': 4}]
