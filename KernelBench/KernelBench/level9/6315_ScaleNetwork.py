import torch
import torch.nn as nn


class ScaleNetwork(nn.Module):
    """Network for parameterizing a scaling function"""

    def __init__(self, input_dim):
        super(ScaleNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2000)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2000, 2000)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(2000, 2000)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(2000, 400)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(400, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.fc5(x).exp()
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
