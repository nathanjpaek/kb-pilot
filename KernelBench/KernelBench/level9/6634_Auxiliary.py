import torch
import torch.nn as nn
import torch.nn.functional as F


class Auxiliary(nn.Module):

    def __init__(self, input_channels, n_classes):
        super(Auxiliary, self).__init__()
        self.Conv2 = nn.Conv2d(input_channels, 128, kernel_size=1)
        self.FC1 = nn.Linear(2048, 1024)
        self.FC2 = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.Conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.FC1(x), inplace=True)
        x = F.dropout(x, 0.7, training=self.training)
        x = self.FC2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'n_classes': 4}]
