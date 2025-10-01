import torch
import torch.nn.functional as F


class myDecoder(torch.nn.Module):

    def __init__(self, fomSize, romSize):
        super(myDecoder, self).__init__()
        self.romSize_ = romSize
        self.fomSize_ = fomSize
        self.fc1 = torch.nn.Linear(romSize, 64)
        self.fc2 = torch.nn.Linear(64, 200)
        self.fc3 = torch.nn.Linear(200, fomSize)

    def forward(self, x):
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'fomSize': 4, 'romSize': 4}]
