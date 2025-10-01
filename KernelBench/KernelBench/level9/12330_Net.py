import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_drop = nn.Dropout2d()
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(7 * 7 * 128, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv4_drop(self.conv4(x)), kernel_size
            =2, stride=2))
        x = F.relu(F.max_pool2d(self.conv5_drop(self.conv5(x)), kernel_size
            =2, stride=2))
        x = x.view(-1, 7 * 7 * 128)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def get_inputs():
    return [torch.rand([4, 3, 243, 243])]


def get_init_inputs():
    return [[], {}]
