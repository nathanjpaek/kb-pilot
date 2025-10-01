import torch
from torch import nn
import torch.nn.functional as F


class ImagePairEncoderV2(nn.Module):

    def __init__(self, init_scale=1.0, bias=True, no_weight_init=False):
        super(ImagePairEncoderV2, self).__init__()
        self.conv1 = nn.Conv2d(9, 64, kernel_size=5, stride=2, bias=bias)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, bias=bias)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, bias=bias)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=1, bias=bias)
        if not no_weight_init:
            for layer in (self.conv1, self.conv2, self.conv3, self.conv4):
                nn.init.orthogonal_(layer.weight, init_scale)

    def forward(self, src_imgs, dst_imgs):
        imgs = torch.cat([src_imgs, dst_imgs, src_imgs - dst_imgs], dim=1)
        x = F.relu(self.conv1(imgs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x.view(x.size(0), -1)


def get_inputs():
    return [torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
