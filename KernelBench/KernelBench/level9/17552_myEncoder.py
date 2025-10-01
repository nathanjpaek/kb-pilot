import torch
import torch.nn.functional as F


class myEncoder(torch.nn.Module):

    def __init__(self, fomSize, romSize):
        super(myEncoder, self).__init__()
        self.fc1 = torch.nn.Linear(fomSize, 200)
        self.fc2 = torch.nn.Linear(200, 64)
        self.fc3 = torch.nn.Linear(64, romSize)

    def forward(self, x):
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        x = F.elu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'fomSize': 4, 'romSize': 4}]
