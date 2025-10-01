from _paritybench_helpers import _mock_config
import torch
import torch.utils.data
import torch
from torchvision.transforms import functional as F
from torch import nn
import torch.nn.functional as F


class EmbedNet(nn.Module):

    def __init__(self, cfg):
        super(EmbedNet, self).__init__()
        self.embed_conv1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
        self.embed_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1,
            padding=1)
        self.embed_conv3 = nn.Conv2d(512, 2048, kernel_size=1, stride=1)
        for l in [self.embed_conv1, self.embed_conv2, self.embed_conv3]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.zeros_(l.bias)

    def forward(self, x):
        x = F.relu(self.embed_conv1(x))
        x = F.relu(self.embed_conv2(x))
        x = self.embed_conv3(x)
        return x


def get_inputs():
    return [torch.rand([4, 1024, 64, 64])]


def get_init_inputs():
    return [[], {'cfg': _mock_config()}]
