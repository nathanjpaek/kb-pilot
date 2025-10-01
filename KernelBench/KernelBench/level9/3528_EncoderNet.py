import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class EncoderNet(nn.Module):

    def __init__(self, pedestrian_num, input_size, hidden_size):
        super(EncoderNet, self).__init__()
        self.pedestrian_num = pedestrian_num
        self.input_size = input_size
        self.hidden_size = hidden_size
        hidden1_size = 32
        hidden2_size = 64
        self.fc1 = torch.nn.Linear(input_size, hidden1_size)
        self.fc2 = torch.nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = torch.nn.Linear(hidden2_size, hidden_size)

    def forward(self, input_traces):
        hidden_list = []
        for i in range(self.pedestrian_num):
            input_trace = input_traces[:, i, :]
            hidden_trace = F.relu(self.fc1(input_trace))
            hidden_trace = F.relu(self.fc2(hidden_trace))
            hidden_trace = self.fc3(hidden_trace)
            hidden_list.append(hidden_trace)
        hidden_traces = torch.stack(hidden_list, 1)
        return hidden_traces


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'pedestrian_num': 4, 'input_size': 4, 'hidden_size': 4}]
