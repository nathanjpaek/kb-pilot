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


class MLP_g(nn.Module):

    def __init__(self, opt):
        super(MLP_g, self).__init__()
        self.SharedFC = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc1 = nn.Linear(opt.ngh, opt.resSize)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, atts, attu):
        hs = torch.cat((noise, atts), 1)
        hu = torch.cat((noise, attu), 1)
        hs = self.lrelu(self.SharedFC(hs))
        hu = self.lrelu(self.SharedFC(hu))
        hs = self.lrelu(self.fc1(hs))
        hu = self.lrelu(self.fc2(hu))
        return hs, hu


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config(attSize=4, nz=4, ngh=4, resSize=4)}]
