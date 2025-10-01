import torch
import torch.utils.data
import torch
import torch.nn as nn


class LittleAdversarialNetwork(nn.Module):

    def __init__(self, in_feature):
        super(LittleAdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.sigmoid(x)
        return x

    def output_num(self):
        return 1


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_feature': 4}]
