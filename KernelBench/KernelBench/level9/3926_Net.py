import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=None, barcode_dim=0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = [250, 100]
        self.fc1 = nn.Linear(input_dim, hidden_dim[0])
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], output_dim)
        self.barcode_dim = barcode_dim

    def forward(self, x, bar):
        x = x.view((x.size(0), -1))
        bar = bar.view((bar.size(0), -1))
        if self.barcode_dim > 0:
            x = torch.cat((x, bar), dim=1)
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        out = self.act(out)
        out = self.fc3(out)
        return out


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
