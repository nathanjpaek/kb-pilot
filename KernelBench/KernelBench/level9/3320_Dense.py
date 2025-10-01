import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense(nn.Module):

    def __init__(self, in_features):
        super(Dense, self).__init__()
        self.fc1 = nn.Linear(in_features, 152)
        self.fc2 = nn.Linear(152, 48)
        self.fc3 = nn.Linear(48, 1)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)

    def maximization(self):
        for param in self.fc1.parameters():
            param.requires_grad_()
        for param in self.fc2.parameters():
            param.requires_grad_()
        for param in self.fc3.parameters():
            param.requires_grad_()

    def expectation(self):
        for param in self.fc1.parameters():
            param.requires_grad_(False)
        for param in self.fc2.parameters():
            param.requires_grad_(False)
        for param in self.fc3.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4}]
