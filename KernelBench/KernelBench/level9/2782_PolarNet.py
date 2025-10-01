import torch
import torch.utils.data
import torch.nn as nn


class PolarNet(torch.nn.Module):

    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.layer1 = nn.Linear(2, num_hid)
        self.layer2 = nn.Linear(num_hid, 1)

    def forward(self, input):
        r = torch.sqrt(input[:, 0] * input[:, 0] + input[:, 1] * input[:, 1]
            ).view(-1, 1)
        a = torch.atan2(input[:, 1], input[:, 0]).view(-1, 1)
        input_polar = torch.cat((r, a), 1).view(-1, 2)
        self.hid1 = torch.tanh(self.layer1(input_polar))
        output = self.layer2(self.hid1)
        output = torch.sigmoid(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_hid': 4}]
