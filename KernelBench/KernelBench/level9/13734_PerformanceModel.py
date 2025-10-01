import torch
import torch.nn as nn


class PerformanceModel(nn.Module):

    def __init__(self, input_len):
        super(PerformanceModel, self).__init__()
        self.input_len = input_len
        self.linear1 = nn.Linear(self.input_len, 32, bias=True)
        self.dropout1 = nn.Dropout(p=0.01)
        self.activate1 = torch.relu
        self.linear2 = nn.Linear(32, 64, bias=True)
        self.dropout2 = nn.Dropout(p=0.01)
        self.activate2 = torch.relu
        self.linear3 = nn.Linear(64, 128, bias=True)
        self.dropout3 = nn.Dropout(p=0.01)
        self.activate3 = torch.relu
        self.linear4 = nn.Linear(128, 64, bias=True)
        self.dropout4 = nn.Dropout(p=0.01)
        self.activate4 = torch.relu
        self.linear5 = nn.Linear(64, 16, bias=True)
        self.activate5 = torch.relu
        self.linear6 = nn.Linear(16, 1, bias=True)
        self.activate6 = torch.relu

    def forward(self, inputs):
        output1 = self.activate1(self.dropout1(self.linear1(inputs)))
        output2 = self.activate2(self.dropout2(self.linear2(output1)))
        output3 = self.activate3(self.dropout3(self.linear3(output2)))
        output4 = self.activate4(self.dropout4(self.linear4(output3)))
        output5 = self.activate5(self.linear5(output4))
        output6 = self.activate6(self.linear6(output5))
        return output6


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_len': 4}]
