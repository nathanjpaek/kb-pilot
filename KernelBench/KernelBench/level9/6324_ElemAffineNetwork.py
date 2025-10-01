import torch
import torch.nn as nn


class ElemAffineNetwork(nn.Module):
    """Network for parameterizing affine transformation"""

    def __init__(self, input_dim):
        super(ElemAffineNetwork, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 2000)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2000, 2000)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(2000, 2000)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(2000, 2000)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(2000, 2 * input_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        scale = torch.exp(x[:, :self.input_dim // 2])
        shift = torch.tanh(x[:, self.input_dim // 2:])
        return scale, shift


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
