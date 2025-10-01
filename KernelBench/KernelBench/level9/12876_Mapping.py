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


def reparameter(mu, sigma):
    return torch.randn_like(mu) * sigma + mu


class Mapping(nn.Module):

    def __init__(self, opt):
        super(Mapping, self).__init__()
        self.latensize = opt.latenSize
        self.encoder_linear = nn.Linear(opt.resSize, opt.latenSize * 2)
        self.discriminator = nn.Linear(opt.latenSize, 1)
        self.classifier = nn.Linear(opt.latenSize, opt.nclass_seen)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, x, train_G=False):
        laten = self.lrelu(self.encoder_linear(x))
        mus, stds = laten[:, :self.latensize], laten[:, self.latensize:]
        stds = self.sigmoid(stds)
        encoder_out = reparameter(mus, stds)
        if not train_G:
            dis_out = self.discriminator(encoder_out)
        else:
            dis_out = self.discriminator(mus)
        pred = self.logic(self.classifier(mus))
        return mus, stds, dis_out, pred, encoder_out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config(latenSize=4, resSize=4, nclass_seen=4)}]
