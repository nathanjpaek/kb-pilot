import torch
import torch.nn as nn


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DenseNet_conv(nn.Module):
    """
    doc
    """

    def __init__(self, in_c, L=5, k=12, bn=False):
        """
        dense block
        :param in_c: input channel number
        :param L: layer number in dense block
        :param k: output channels of each layer in dense block
        :param bn: using bn or not
        """
        super(DenseNet_conv, self).__init__()
        self.L = L
        self.k = k
        self.bn = bn
        self.conv1s = []
        self.conv2s = []
        self.bn1s = []
        self.bn2s = []
        for i in range(self.L):
            channel_in = i * self.k + in_c + 2
            conv1 = nn.Conv2d(channel_in, self.k * 4, kernel_size=1, stride=1)
            setattr(self, 'conv1_%i' % i, conv1)
            xavier_init(conv1)
            self.conv1s.append(conv1)
            if self.bn:
                bn1 = nn.BatchNorm2d(num_features=self.k * 4)
                setattr(self, 'bn1_%i' % i, bn1)
                self.bn1s.append(bn1)
            conv2 = nn.Conv2d(self.k * 4, self.k, kernel_size=3, stride=1,
                padding=1)
            setattr(self, 'conv2_%i' % i, conv2)
            xavier_init(conv2)
            self.conv2s.append(conv2)
            if self.bn:
                bn2 = nn.BatchNorm2d(num_features=self.k)
                setattr(self, 'bn2_%i' % i, bn2)
                self.bn2s.append(bn2)

    def forward(self, x, sparse_inputs):
        """
        dense block
        :param x: x
        :param sparse_inputs: sparse image (s1,s2), 2 channels
        :return:
        """
        hs = []
        h = torch.cat((x, sparse_inputs), 1)
        hs.append(h)
        for i in range(self.L):
            if i != 0:
                h = torch.cat(hs, 1)
            h = self.conv1s[i](h)
            if self.bn:
                h = self.bn1s[i](h)
            h = torch.relu(h)
            h = self.conv2s[i](h)
            if self.bn:
                h = self.bn2s[i](h)
            h = torch.relu(h)
            if i != self.L - 1:
                hs.append(h)
        return h


def get_inputs():
    return [torch.rand([4, 2, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_c': 4}]
