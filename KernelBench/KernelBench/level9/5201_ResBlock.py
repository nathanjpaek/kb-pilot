import torch
from torch import nn
import torch.utils.data
import torch.autograd


class ResBlock(nn.Module):

    def __init__(self, num_features, use_batch_norm=False):
        super(ResBlock, self).__init__()
        self.num_features = num_features
        self.conv_layer1 = nn.Conv2d(num_features, num_features,
            kernel_size=3, stride=1, padding=1)
        self.relu_layer = nn.ReLU()
        self.conv_layer2 = nn.Conv2d(num_features, num_features,
            kernel_size=3, stride=1, padding=1)
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.batch_norm_layer1 = nn.BatchNorm2d(self.num_features)
            self.batch_norm_layer2 = nn.BatchNorm2d(self.num_features)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        residual = x
        x = self.conv_layer1(x)
        if self.use_batch_norm:
            x = self.batch_norm_layer1(x)
        x = self.relu_layer(x)
        x = self.conv_layer2(x)
        if self.use_batch_norm:
            x = self.batch_norm_layer2(x)
        x += residual
        x = self.relu_layer(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
