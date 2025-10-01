from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class ResnetQ(nn.Module):

    def __init__(self, opt):
        super(ResnetQ, self).__init__()
        self.conv = nn.Linear(opt.ndf, opt.ndf)
        self.lReLU = nn.LeakyReLU(0.1, inplace=True)
        self.conv_disc = nn.Linear(opt.ndf, 10)
        self.conv_mu = nn.Linear(opt.ndf, 2)
        self.conv_var = nn.Linear(opt.ndf, 2)

    def forward(self, x):
        y = self.conv(x)
        disc_logits = self.conv_disc(y).squeeze()
        mu = self.conv_mu(y).squeeze()
        var = self.conv_var(y).squeeze().exp()
        return disc_logits, mu, var


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config(ndf=4)}]
