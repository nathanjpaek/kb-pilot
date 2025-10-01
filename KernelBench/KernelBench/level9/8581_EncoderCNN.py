import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):

    def __init__(self, latent_dim=1024):
        super(EncoderCNN, self).__init__()
        self.latent_dim = latent_dim
        self.conv1_1 = nn.Conv2d(8, 8, 4, stride=2, dilation=1, padding=1)
        self.conv1_2 = nn.Conv2d(8, 8, 4, stride=2, dilation=2, padding=3)
        self.conv1_4 = nn.Conv2d(8, 8, 4, stride=2, dilation=4, padding=6)
        self.conv1_8 = nn.Conv2d(8, 8, 4, stride=2, dilation=8, padding=12)
        self.conv1_16 = nn.Conv2d(8, 8, 4, stride=2, dilation=16, padding=24)
        self.conv2 = nn.Conv2d(8 * 5, self.latent_dim // 4, 4, stride=2,
            padding=1)
        self.conv3 = nn.Conv2d(self.latent_dim // 4, self.latent_dim // 2, 
            4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(self.latent_dim // 2, self.latent_dim, 4,
            stride=2, padding=1)

    def forward(self, x):
        x1 = F.elu(self.conv1_1(x))
        x2 = F.elu(self.conv1_2(x))
        x4 = F.elu(self.conv1_4(x))
        x8 = F.elu(self.conv1_8(x))
        x16 = F.elu(self.conv1_16(x))
        x = torch.cat((x1, x2, x4, x8, x16), dim=1)
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.avg_pool2d(x, 2).squeeze()
        return x

    def train_order_block_ids(self):
        return [[0, 9], [10, 15]]


def get_inputs():
    return [torch.rand([4, 8, 64, 64])]


def get_init_inputs():
    return [[], {}]
