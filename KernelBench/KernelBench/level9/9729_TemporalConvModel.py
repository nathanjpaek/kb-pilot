import torch
import torch.nn as nn


class TemporalConvModel(nn.Module):

    def __init__(self, in_feature, seq_len):
        super(TemporalConvModel, self).__init__()
        self.conv1 = nn.Conv1d(in_feature, 256, 1, 1)
        self.conv2 = nn.Conv1d(256, 256, 3, 1, 1)
        self.conv3 = nn.Conv1d(256, 256, 3, 1, 1)
        self.fc = nn.Linear(256 * seq_len, 2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.seq_len = seq_len

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(-1, 256 * self.seq_len)
        x = self.fc(x)
        return x

    def output_num(self):
        return 1


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_feature': 4, 'seq_len': 4}]
