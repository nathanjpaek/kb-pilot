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


class GaussianBlock(nn.Module):

    def __init__(self, in_features, n_z):
        """
        :param input_feature: num of input feature
        :param n_z: dim of distribution
        """
        super(GaussianBlock, self).__init__()
        self.n_x = in_features
        self.n_z = n_z
        self.z_mu_fc = FullyConnected(self.n_x, self.n_z)
        self.z_log_var_fc = FullyConnected(self.n_x, self.n_z)

    def forward(self, x):
        y = x
        mu = self.z_mu_fc(y)
        log_var = self.z_log_var_fc(y)
        log_var = torch.clamp(log_var, min=-20.0, max=3.0)
        return mu, log_var


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'n_z': 4}]
