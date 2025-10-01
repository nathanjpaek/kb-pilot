import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class FeatureDiscriminator(nn.Module):

    def __init__(self):
        super(FeatureDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=256,
            kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128,
            kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64,
            kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size
            =1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=
            1, stride=1)
        self.sig5 = nn.Sigmoid()
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 25), stride=1)
        self._do_initializer()

    def _do_initializer(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(tensor=module.weight, mean=0, std=0.01)

    def forward(self, inputs):
        out = F.leaky_relu(self.conv1(inputs))
        out = F.leaky_relu(self.conv2(out))
        out = F.leaky_relu(self.conv3(out))
        out = F.leaky_relu(self.conv4(out))
        out = self.sig5(self.conv5(out))
        out = self.avg_pool(out)
        out = out.view(-1)
        return out


def get_inputs():
    return [torch.rand([4, 512, 64, 64])]


def get_init_inputs():
    return [[], {}]
