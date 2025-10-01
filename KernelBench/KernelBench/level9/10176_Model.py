import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 60, kernel_size=5)
        self.conv2 = nn.Conv2d(60, 60, kernel_size=5)
        self.conv3 = nn.Conv2d(60, 30, kernel_size=3)
        self.conv4 = nn.Conv2d(30, 30, kernel_size=3)
        self.lin1 = nn.Linear(4 * 4 * 30, 500)
        self.lin2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 4 * 4 * 30)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
