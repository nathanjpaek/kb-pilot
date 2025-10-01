import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCNN2(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_names = ['conv11', 'conv12', 'conv21', 'conv22',
            'conv31', 'conv32', 'fc1', 'output_layer']
        self.conv11 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv12 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv21 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv22 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv31 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv32 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4 * 4 * 128, 128)
        self.output_layer = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.pool(F.relu(self.conv12(x)))
        x = self.dropout(x)
        x = F.relu(self.conv21(x))
        x = self.pool(F.relu(self.conv22(x)))
        x = self.dropout(x)
        x = F.relu(self.conv31(x))
        x = self.pool(F.relu(self.conv32(x)))
        x = self.dropout(x)
        x = x.view(-1, 4 * 4 * 128)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
