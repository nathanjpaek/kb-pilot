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


class AttDec(nn.Module):

    def __init__(self, opt, attSize):
        super(AttDec, self).__init__()
        self.embedSz = 0
        self.fc1 = nn.Linear(opt.resSize + self.embedSz, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.hidden = None
        self.sigmoid = None
        self.apply(weights_init)

    def forward(self, feat, att=None):
        h = feat
        if self.embedSz > 0:
            assert att is not None, 'Conditional Decoder requires attribute input'
            h = torch.cat((feat, att), 1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc3(self.hidden)
        if self.sigmoid is not None:
            h = self.sigmoid(h)
        else:
            h = h / h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0), h
                .size(1))
        self.out = h
        return h

    def getLayersOutDet(self):
        return self.hidden.detach()


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config(resSize=4, ngh=4), 'attSize': 4}]
