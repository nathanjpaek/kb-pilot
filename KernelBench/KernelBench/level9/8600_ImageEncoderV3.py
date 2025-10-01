import torch
from torch import nn
import torch.nn.functional as F


class ImageEncoderV3(nn.Module):

    def __init__(self, output_dim=512, init_scale=1.0, residual_link=False):
        super(ImageEncoderV3, self).__init__()
        self.residual_link = residual_link
        self.conv1 = nn.Conv2d(3, output_dim // 8, kernel_size=5, stride=2)
        if residual_link:
            self.res_fc1 = nn.Conv2d(output_dim // 8, output_dim // 4,
                kernel_size=1, stride=2)
        self.conv2 = nn.Conv2d(output_dim // 8, output_dim // 4,
            kernel_size=5, stride=2)
        if residual_link:
            self.res_fc2 = nn.Conv2d(output_dim // 4, output_dim // 2,
                kernel_size=1, stride=2)
        self.conv3 = nn.Conv2d(output_dim // 4, output_dim // 2,
            kernel_size=5, stride=2)
        if residual_link:
            self.res_fc3 = nn.Conv2d(output_dim // 2, output_dim,
                kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(output_dim // 2, output_dim, kernel_size=5,
            stride=1)
        for layer in (self.conv1, self.conv2, self.conv3, self.conv4):
            nn.init.orthogonal_(layer.weight, init_scale)

    def forward(self, imgs):
        if self.residual_link:
            x = F.relu(self.conv1(imgs))
            x = F.relu(self.res_fc1(x[:, :, 2:-2, 2:-2]) + self.conv2(x))
            x = F.relu(self.res_fc2(x[:, :, 2:-2, 2:-2]) + self.conv3(x))
            x = F.relu(self.res_fc3(x[:, :, 2:-2, 2:-2]) + self.conv4(x))
        else:
            x = F.relu(self.conv1(imgs))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
        return x.view(x.size(0), -1)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
