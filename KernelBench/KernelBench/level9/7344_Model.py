import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3
            )
        self.pool1_3x3_s2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.pool1_norm1 = nn.LocalResponseNorm(2, 1.99999994948e-05, 0.75)
        self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=1, stride=1,
            padding=0)
        self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.conv2_norm2 = nn.LocalResponseNorm(2, 1.99999994948e-05, 0.75)
        self.pool2_3x3_s2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=1, stride=1,
            padding=0)
        self.inception_3a_3x3_reduce = nn.Conv2d(192, 96, kernel_size=1,
            stride=1, padding=0)
        self.inception_3a_3x3 = nn.Conv2d(96, 128, kernel_size=3, stride=1,
            padding=1)
        self.inception_3a_5x5_reduce = nn.Conv2d(192, 16, kernel_size=1,
            stride=1, padding=0)
        self.inception_3a_5x5 = nn.Conv2d(16, 32, kernel_size=5, stride=1,
            padding=2)
        self.inception_3a_pool = nn.MaxPool2d(3, stride=1, padding=1,
            ceil_mode=True)
        self.inception_3a_pool_proj = nn.Conv2d(192, 32, kernel_size=1,
            stride=1, padding=0)
        self.inception_3b_1x1 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
            padding=0)
        self.inception_3b_3x3_reduce = nn.Conv2d(256, 128, kernel_size=1,
            stride=1, padding=0)
        self.inception_3b_3x3 = nn.Conv2d(128, 192, kernel_size=3, stride=1,
            padding=1)
        self.inception_3b_5x5_reduce = nn.Conv2d(256, 32, kernel_size=1,
            stride=1, padding=0)
        self.inception_3b_5x5 = nn.Conv2d(32, 96, kernel_size=5, stride=1,
            padding=2)
        self.inception_3b_pool = nn.MaxPool2d(3, stride=1, padding=1,
            ceil_mode=True)
        self.inception_3b_pool_proj = nn.Conv2d(256, 64, kernel_size=1,
            stride=1, padding=0)
        self.pool3_3x3_s2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception_4a_1x1 = nn.Conv2d(480, 192, kernel_size=1, stride=1,
            padding=0)
        self.inception_4a_3x3_reduce = nn.Conv2d(480, 96, kernel_size=1,
            stride=1, padding=0)
        self.inception_4a_3x3 = nn.Conv2d(96, 208, kernel_size=3, stride=1,
            padding=1)
        self.inception_4a_5x5_reduce = nn.Conv2d(480, 16, kernel_size=1,
            stride=1, padding=0)
        self.inception_4a_5x5 = nn.Conv2d(16, 48, kernel_size=5, stride=1,
            padding=2)
        self.inception_4a_pool = nn.MaxPool2d(3, stride=1, padding=1,
            ceil_mode=True)
        self.inception_4a_pool_proj = nn.Conv2d(480, 64, kernel_size=1,
            stride=1, padding=0)
        self.inception_4b_1x1 = nn.Conv2d(512, 160, kernel_size=1, stride=1,
            padding=0)
        self.inception_4b_3x3_reduce = nn.Conv2d(512, 112, kernel_size=1,
            stride=1, padding=0)
        self.inception_4b_3x3 = nn.Conv2d(112, 224, kernel_size=3, stride=1,
            padding=1)
        self.inception_4b_5x5_reduce = nn.Conv2d(512, 24, kernel_size=1,
            stride=1, padding=0)
        self.inception_4b_5x5 = nn.Conv2d(24, 64, kernel_size=5, stride=1,
            padding=2)
        self.inception_4b_pool = nn.MaxPool2d(3, stride=1, padding=1,
            ceil_mode=True)
        self.inception_4b_pool_proj = nn.Conv2d(512, 64, kernel_size=1,
            stride=1, padding=0)
        self.inception_4c_1x1 = nn.Conv2d(512, 128, kernel_size=1, stride=1,
            padding=0)
        self.inception_4c_3x3_reduce = nn.Conv2d(512, 128, kernel_size=1,
            stride=1, padding=0)
        self.inception_4c_3x3 = nn.Conv2d(128, 256, kernel_size=3, stride=1,
            padding=1)
        self.inception_4c_5x5_reduce = nn.Conv2d(512, 24, kernel_size=1,
            stride=1, padding=0)
        self.inception_4c_5x5 = nn.Conv2d(24, 64, kernel_size=5, stride=1,
            padding=2)
        self.inception_4c_pool = nn.MaxPool2d(3, stride=1, padding=1,
            ceil_mode=True)
        self.inception_4c_pool_proj = nn.Conv2d(512, 64, kernel_size=1,
            stride=1, padding=0)
        self.inception_4d_1x1 = nn.Conv2d(512, 112, kernel_size=1, stride=1,
            padding=0)
        self.inception_4d_3x3_reduce = nn.Conv2d(512, 144, kernel_size=1,
            stride=1, padding=0)
        self.inception_4d_3x3 = nn.Conv2d(144, 288, kernel_size=3, stride=1,
            padding=1)
        self.inception_4d_5x5_reduce = nn.Conv2d(512, 32, kernel_size=1,
            stride=1, padding=0)
        self.inception_4d_5x5 = nn.Conv2d(32, 64, kernel_size=5, stride=1,
            padding=2)
        self.inception_4d_pool = nn.MaxPool2d(3, stride=1, padding=1,
            ceil_mode=True)
        self.inception_4d_pool_proj = nn.Conv2d(512, 64, kernel_size=1,
            stride=1, padding=0)
        self.inception_4e_1x1 = nn.Conv2d(528, 256, kernel_size=1, stride=1,
            padding=0)
        self.inception_4e_3x3_reduce = nn.Conv2d(528, 160, kernel_size=1,
            stride=1, padding=0)
        self.inception_4e_3x3 = nn.Conv2d(160, 320, kernel_size=3, stride=1,
            padding=1)
        self.inception_4e_5x5_reduce = nn.Conv2d(528, 32, kernel_size=1,
            stride=1, padding=0)
        self.inception_4e_5x5 = nn.Conv2d(32, 128, kernel_size=5, stride=1,
            padding=2)
        self.inception_4e_pool = nn.MaxPool2d(3, stride=1, padding=1,
            ceil_mode=True)
        self.inception_4e_pool_proj = nn.Conv2d(528, 128, kernel_size=1,
            stride=1, padding=0)
        self.pool4_3x3_s2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception_5a_1x1 = nn.Conv2d(832, 256, kernel_size=1, stride=1,
            padding=0)
        self.inception_5a_3x3_reduce = nn.Conv2d(832, 160, kernel_size=1,
            stride=1, padding=0)
        self.inception_5a_3x3 = nn.Conv2d(160, 320, kernel_size=3, stride=1,
            padding=1)
        self.inception_5a_5x5_reduce = nn.Conv2d(832, 32, kernel_size=1,
            stride=1, padding=0)
        self.inception_5a_5x5 = nn.Conv2d(32, 128, kernel_size=5, stride=1,
            padding=2)
        self.inception_5a_pool = nn.MaxPool2d(3, stride=1, padding=1,
            ceil_mode=True)
        self.inception_5a_pool_proj = nn.Conv2d(832, 128, kernel_size=1,
            stride=1, padding=0)
        self.inception_5b_1x1 = nn.Conv2d(832, 384, kernel_size=1, stride=1,
            padding=0)
        self.inception_5b_3x3_reduce = nn.Conv2d(832, 192, kernel_size=1,
            stride=1, padding=0)
        self.inception_5b_3x3 = nn.Conv2d(192, 384, kernel_size=3, stride=1,
            padding=1)
        self.inception_5b_5x5_reduce = nn.Conv2d(832, 48, kernel_size=1,
            stride=1, padding=0)
        self.inception_5b_5x5 = nn.Conv2d(48, 128, kernel_size=5, stride=1,
            padding=2)
        self.inception_5b_pool = nn.MaxPool2d(3, stride=1, padding=1,
            ceil_mode=True)
        self.inception_5b_pool_proj = nn.Conv2d(832, 128, kernel_size=1,
            stride=1, padding=0)
        self.pool5_7x7_s1 = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.loss3_SLclassifier = nn.Linear(1024, 61)

    def forward(self, x):
        x = F.relu(self.conv1_7x7_s2(x))
        x = self.pool1_3x3_s2(x)
        x = self.pool1_norm1(x)
        x = F.relu(self.conv2_3x3_reduce(x))
        x = F.relu(self.conv2_3x3(x))
        x = self.conv2_norm2(x)
        x = self.pool2_3x3_s2(x)
        x1 = F.relu(self.inception_3a_1x1(x))
        x2 = F.relu(self.inception_3a_3x3_reduce(x))
        x2 = F.relu(self.inception_3a_3x3(x2))
        x3 = F.relu(self.inception_3a_5x5_reduce(x))
        x3 = F.relu(self.inception_3a_5x5(x3))
        x4 = self.inception_3a_pool(x)
        x4 = F.relu(self.inception_3a_pool_proj(x4))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x1 = F.relu(self.inception_3b_1x1(x))
        x2 = F.relu(self.inception_3b_3x3_reduce(x))
        x2 = F.relu(self.inception_3b_3x3(x2))
        x3 = F.relu(self.inception_3b_5x5_reduce(x))
        x3 = F.relu(self.inception_3b_5x5(x3))
        x4 = self.inception_3b_pool(x)
        x4 = F.relu(self.inception_3b_pool_proj(x4))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.pool3_3x3_s2(x)
        x1 = F.relu(self.inception_4a_1x1(x))
        x2 = F.relu(self.inception_4a_3x3_reduce(x))
        x2 = F.relu(self.inception_4a_3x3(x2))
        x3 = F.relu(self.inception_4a_5x5_reduce(x))
        x3 = F.relu(self.inception_4a_5x5(x3))
        x4 = self.inception_4a_pool(x)
        x4 = F.relu(self.inception_4a_pool_proj(x4))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x1 = F.relu(self.inception_4b_1x1(x))
        x2 = F.relu(self.inception_4b_3x3_reduce(x))
        x2 = F.relu(self.inception_4b_3x3(x2))
        x3 = F.relu(self.inception_4b_5x5_reduce(x))
        x3 = F.relu(self.inception_4b_5x5(x3))
        x4 = self.inception_4b_pool(x)
        x4 = F.relu(self.inception_4b_pool_proj(x4))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x1 = F.relu(self.inception_4c_1x1(x))
        x2 = F.relu(self.inception_4c_3x3_reduce(x))
        x2 = F.relu(self.inception_4c_3x3(x2))
        x3 = F.relu(self.inception_4c_5x5_reduce(x))
        x3 = F.relu(self.inception_4c_5x5(x3))
        x4 = self.inception_4c_pool(x)
        x4 = F.relu(self.inception_4c_pool_proj(x4))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x1 = F.relu(self.inception_4d_1x1(x))
        x2 = F.relu(self.inception_4d_3x3_reduce(x))
        x2 = F.relu(self.inception_4d_3x3(x2))
        x3 = F.relu(self.inception_4d_5x5_reduce(x))
        x3 = F.relu(self.inception_4d_5x5(x3))
        x4 = self.inception_4d_pool(x)
        x4 = F.relu(self.inception_4d_pool_proj(x4))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x1 = F.relu(self.inception_4e_1x1(x))
        x2 = F.relu(self.inception_4e_3x3_reduce(x))
        x2 = F.relu(self.inception_4e_3x3(x2))
        x3 = F.relu(self.inception_4e_5x5_reduce(x))
        x3 = F.relu(self.inception_4e_5x5(x3))
        x4 = self.inception_4e_pool(x)
        x4 = F.relu(self.inception_4e_pool_proj(x4))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.pool4_3x3_s2(x)
        x1 = F.relu(self.inception_5a_1x1(x))
        x2 = F.relu(self.inception_5a_3x3_reduce(x))
        x2 = F.relu(self.inception_5a_3x3(x2))
        x3 = F.relu(self.inception_5a_5x5_reduce(x))
        x3 = F.relu(self.inception_5a_5x5(x3))
        x4 = self.inception_5a_pool(x)
        x4 = F.relu(self.inception_5a_pool_proj(x4))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x1 = F.relu(self.inception_5b_1x1(x))
        x2 = F.relu(self.inception_5b_3x3_reduce(x))
        x2 = F.relu(self.inception_5b_3x3(x2))
        x3 = F.relu(self.inception_5b_5x5_reduce(x))
        x3 = F.relu(self.inception_5b_5x5(x3))
        x4 = self.inception_5b_pool(x)
        x4 = F.relu(self.inception_5b_pool_proj(x4))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.pool5_7x7_s1(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.loss3_SLclassifier(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
