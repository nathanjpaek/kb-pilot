import math
import torch
import torch.nn as nn
import torch.optim
import torch.multiprocessing
from torch.nn.parameter import Parameter


class FullyConnected(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        """
        Fully connected layer of learnable weights with learnable bias
        :param self:
        :param in_features: number neurons in
        :param out_features: num neurons out
        :param bias: to use bias (boole)
        :return:
        """
        super(FullyConnected, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class FC_Block(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.LeakyReLU(
        0.1), batch_norm=False, p_dropout=0.0, bias=True):
        """
        Define a fully connected block
        """
        super(FC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_f = activation
        self.batch_norm = batch_norm
        self.p_dropout = p_dropout
        self.do = nn.Dropout(p_dropout)
        self.fc = FullyConnected(self.in_features, self.out_features, bias=bias
            )
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        y = self.fc(x)
        y = self.act_f(y)
        if self.batch_norm:
            b, f = y.shape
            y = self.bn(y.view(b, -1)).view(b, f)
        y = self.do(y)
        return y

    def __repr__(self):
        representation = self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')' + ', dropout: ' + str(
            self.p_dropout)
        if self.batch_norm:
            representation = representation + ', batch norm'
        representation = representation + ', act_fn: {0}'.format(self.act_f)
        return representation


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
