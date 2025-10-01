from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.utils.data


class AE(nn.Module):
    """ Class for the AE using Fully Connected 
    """

    def __init__(self, opt):
        super().__init__()
        assert opt.isize % 4 == 0, 'input size has to be a multiple of 4'
        self.dense1 = nn.Linear(opt.isize, opt.isize)
        self.dense2 = nn.Linear(int(opt.isize / 2), int(opt.isize / 2))
        self.dense3 = nn.Linear(int(opt.isize / 4), int(opt.isize / 4))
        self.pool = nn.MaxPool1d(2, padding=0)
        self.up = nn.Upsample(scale_factor=2)
        self.dense4 = nn.Linear(int(opt.isize / 2), int(opt.isize / 2))
        self.dense5 = nn.Linear(opt.isize, opt.isize)

    def forward(self, x):
        x1 = self.dense1(x)
        x1 = nn.ReLU()(x1)
        x2 = self.pool(torch.unsqueeze(x1, 0))
        x2 = torch.squeeze(x2, 0)
        x2 = self.dense2(x2)
        x2 = nn.ReLU()(x2)
        x3 = self.pool(torch.unsqueeze(x2, 0))
        x3 = torch.squeeze(x3, 0)
        encoded = self.dense3(x3)
        encoded = nn.ReLU()(encoded)
        y = self.up(torch.unsqueeze(encoded, 1))
        y = torch.squeeze(y, 1)
        y = x2 + y
        y = self.dense4(y)
        y = nn.ReLU()(y)
        y = self.up(torch.unsqueeze(y, 1))
        y = torch.squeeze(y, 1)
        decoded = self.dense5(y)
        decoded = nn.Tanh()(decoded)
        return decoded


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config(isize=4)}]
