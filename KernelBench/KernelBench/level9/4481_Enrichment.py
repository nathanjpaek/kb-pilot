import torch
import torch.nn as nn


class Enrichment(nn.Module):

    def __init__(self, c_in, rate=2):
        super(Enrichment, self).__init__()
        self.rate = rate
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        dilation = self.rate * 1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation,
            padding=dilation)
        dilation = self.rate * 2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation,
            padding=dilation)
        dilation = self.rate * 3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation,
            padding=dilation)
        dilation = self.rate * 4 if self.rate >= 1 else 1
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation,
            padding=dilation)
        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu(self.conv1(o))
        o2 = self.relu(self.conv2(o))
        o3 = self.relu(self.conv3(o))
        o4 = self.relu(self.conv4(o))
        out = o + o1 + o2 + o3 + o4
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'c_in': 4}]
