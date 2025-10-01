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


class Discriminator(nn.Module):

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.discriminatorS = nn.Linear(opt.attSize, 256)
        self.discriminatorU = nn.Linear(opt.attSize, 256)
        self.fc = nn.Linear(256 * 2, 2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, s, u):
        dis_s = self.lrelu(self.discriminatorS(s))
        dis_u = self.lrelu(self.discriminatorU(u))
        hs = torch.cat((dis_s, dis_u), 1)
        hs = self.fc(hs).squeeze()
        return hs, dis_s, dis_u


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config(attSize=4)}]
