import torch
from torch import nn
import torch.nn.functional as F


class FeatureMapPairEncoderV2(nn.Module):

    def __init__(self, init_scale=1.0, no_weight_init=False):
        super(FeatureMapPairEncoderV2, self).__init__()
        self.conv1 = nn.Conv2d(96, 256, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        if not no_weight_init:
            for layer in (self.conv1, self.conv2):
                nn.init.orthogonal_(layer.weight, init_scale)

    def forward(self, src_imgs, dst_imgs):
        imgs = torch.cat([src_imgs, dst_imgs, src_imgs - dst_imgs], dim=1)
        x = F.relu(self.conv1(imgs))
        x = F.relu(self.conv2(x))
        return x.view(x.size(0), -1)


def get_inputs():
    return [torch.rand([4, 32, 64, 64]), torch.rand([4, 32, 64, 64])]


def get_init_inputs():
    return [[], {}]
