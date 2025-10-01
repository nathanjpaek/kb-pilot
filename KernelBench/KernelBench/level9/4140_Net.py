import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, Cin, Cout):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(Cin, Cout, (3, 3))

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv1(x)
        z = torch.cat([x0, x1])
        output = F.log_softmax(z, dim=1)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'Cin': 4, 'Cout': 4}]
