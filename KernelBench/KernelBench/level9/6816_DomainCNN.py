import torch
from torch.nn import functional as F
import torch.utils.data


class DomainCNN(torch.nn.Module):

    def __init__(self, domains):
        super(DomainCNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 32, kernel_size=5)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=2)
        self.conv2 = torch.nn.Conv1d(32, 16, kernel_size=5, stride=2)
        self.pool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv1d(16, 8, kernel_size=2, stride=2)
        self.pool3 = torch.nn.MaxPool1d(kernel_size=2, stride=1)
        self.fc1 = torch.nn.Linear(8 * 2, domains)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.pool1(x)
        x = F.elu(self.conv2(x))
        x = self.pool2(x)
        x = F.elu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(-1, 8 * 2)
        m = torch.nn.Softmax(1)
        x = m(self.fc1(x))
        return x


def get_inputs():
    return [torch.rand([4, 1, 64])]


def get_init_inputs():
    return [[], {'domains': 4}]
