import torch
import torch.nn as nn


class SmallAdversarialNetwork(nn.Module):

    def __init__(self, in_feature):
        super(SmallAdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 256)
        self.ad_layer2 = nn.Linear(256, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.sigmoid(x)
        return x

    def output_num(self):
        return 1


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_feature': 4}]
