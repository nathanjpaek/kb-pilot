import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class HamidaEtAl(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(HamidaEtAl, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = dilation, 1, 1
        if patch_size == 3:
            self.conv1 = nn.Conv3d(1, 20, (3, 3, 3), stride=(1, 1, 1),
                dilation=dilation, padding=(1, 0, 0))
        else:
            self.conv1 = nn.Conv3d(1, 20, (3, 3, 3), stride=(1, 1, 1),
                dilation=dilation, padding=(1, 0, 0))
        self.pool1 = nn.Conv3d(20, 2, (3, 1, 1), dilation=dilation, stride=
            (2, 1, 1), padding=(1, 0, 0))
        self.conv2 = nn.Conv3d(2, 35, (3, 3, 3), dilation=dilation, stride=
            (1, 1, 1), padding=(1, 0, 0))
        self.pool2 = nn.Conv3d(35, 2, (2, 1, 1), dilation=dilation, stride=
            (2, 1, 1), padding=(1, 0, 0))
        self.conv3 = nn.Conv3d(2, 35, (3, 1, 1), dilation=dilation, stride=
            (1, 1, 1), padding=(1, 0, 0))
        self.pool3 = nn.Conv3d(35, 2, (1, 1, 1), dilation=dilation, stride=
            (2, 1, 1), padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(2, 35, (3, 1, 1), dilation=dilation, stride=
            (1, 1, 1), padding=(1, 0, 0))
        self.pool4 = nn.Conv3d(35, 4, (1, 1, 1), dilation=dilation, stride=
            (2, 2, 2), padding=(0, 0, 0))
        self.dropout = nn.Dropout(p=0.5)
        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, n_classes)
        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels, self.patch_size,
                self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))
            x = self.pool4(self.conv4(x))
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64, 64])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'n_classes': 4}]
