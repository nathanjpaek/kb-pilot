import torch
import torch.nn as nn
import torch.nn.functional as F


class BehaviorClone(nn.Module):

    def __init__(self, input_shape, output_shape):
        super(BehaviorClone, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.fc1 = nn.Linear(input_shape, input_shape // 2)
        self.fc2 = nn.Linear(input_shape // 2, output_shape)
        self.do = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.do(x)
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_shape': 4, 'output_shape': 4}]
