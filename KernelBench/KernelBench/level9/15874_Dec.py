from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Dec(nn.Module):

    def __init__(self, opt):
        super(Dec, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, feat):
        h = self.lrelu(self.fc1(feat))
        h = self.lrelu(self.fc2(h))
        h = self.fc3(h)
        return h


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config(resSize=4, ngh=4, attSize=4)}]
