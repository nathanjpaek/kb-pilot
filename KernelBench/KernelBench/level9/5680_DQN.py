import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        val = int((inputs + outputs) / 2)
        self.fc1 = nn.Linear(inputs, val)
        self.fc2 = nn.Linear(val, val)
        self.fc3 = nn.Linear(val, val)
        self.fc4 = nn.Linear(val, outputs)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inputs': 4, 'outputs': 4}]
