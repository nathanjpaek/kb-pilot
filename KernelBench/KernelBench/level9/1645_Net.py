import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, num_classes=8):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(3, 3))
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv1.bias.data.fill_(0.01)
        self.conv2.bias.data.fill_(0.01)
        self.fc1 = nn.Linear(in_features=18000, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 18000)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 3, 128, 128])]


def get_init_inputs():
    return [[], {}]
