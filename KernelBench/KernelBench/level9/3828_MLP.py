import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Simple MLP to demonstrate Jacobian regularization.
    """

    def __init__(self, in_channel=1, im_size=28, num_classes=10,
        fc_channel1=200, fc_channel2=200):
        super(MLP, self).__init__()
        compression = in_channel * im_size * im_size
        self.compression = compression
        self.fc1 = nn.Linear(compression, fc_channel1)
        self.fc2 = nn.Linear(fc_channel1, fc_channel2)
        self.fc3 = nn.Linear(fc_channel2, num_classes)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = x.view(-1, self.compression)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
