import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data


class DeepTable3(nn.Module):
    """A deep differentialable 'Table' for learning one-hot input and output.
    """

    def __init__(self, in_channels, out_channels, num_hidden1=200,
        num_hidden2=100):
        super(DeepTable3, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_hidden1, bias=False)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2, bias=False)
        self.fc3 = nn.Linear(num_hidden2, out_channels, bias=False)
        self.fc1.weight.data.uniform_(0.0, 0.0)
        self.fc3.weight.data.uniform_(0.0, 0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.softmax(self.fc2(x))
        return self.fc3(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
