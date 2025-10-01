import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class MLP3_hardsig(nn.Module):

    def __init__(self, width=512, p=0.5):
        super(MLP3_hardsig, self).__init__()
        self.fc1 = nn.Linear(32 * 32, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 10)
        self.fc1_out = torch.zeros(1)
        self.do1 = nn.Dropout(p=p)
        self.relu1_out = torch.zeros(1)
        self.fc2_out = torch.zeros(1)
        self.do2 = nn.Dropout(p=p)
        self.relu2_out = torch.zeros(1)
        self.fc3_out = torch.zeros(1)

    def forward(self, x):
        x = x.view(-1, 32 * 32)
        self.fc1_out = self.fc1(x)
        self.relu1_out = nn.Sigmoid()(self.do1(self.fc1_out))
        self.fc2_out = self.fc2(self.relu1_out)
        self.relu2_out = nn.Sigmoid()(self.do2(self.fc2_out))
        self.fc3_out = self.fc3(self.relu2_out)
        return F.softmax(self.fc3_out, dim=1)


def get_inputs():
    return [torch.rand([4, 1024])]


def get_init_inputs():
    return [[], {}]
