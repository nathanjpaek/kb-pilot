import math
import torch
import torch.nn.functional as F
import torch.nn as nn


class DNN(nn.Module):

    def __init__(self, n_concat, freq_bins, *, dropout=0.2):
        super().__init__()
        hidden_units = 2048
        self.dropout = dropout
        self.fc1 = nn.Linear(n_concat * freq_bins, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, freq_bins)
        self.init_weights()

    @staticmethod
    def init_layer(layer):
        """Initialize a Linear or Convolutional layer. 
        Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing 
        human-level performance on imagenet classification." Proceedings of the 
        IEEE international conference on computer vision. 2015.
        """
        if layer.weight.ndimension() == 4:
            _n_out, n_in, height, width = layer.weight.size()
            n = n_in * height * width
        elif layer.weight.ndimension() == 2:
            _n_out, n = layer.weight.size()
        std = math.sqrt(2.0 / n)
        scale = std * math.sqrt(3.0)
        layer.weight.data.uniform_(-scale, scale)
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)

    @staticmethod
    def init_bn(bn):
        """Initialize a Batchnorm layer. """
        bn.bias.data.fill_(0.0)
        bn.weight.data.fill_(1.0)

    def init_weights(self):
        self.init_layer(self.fc1)
        self.init_layer(self.fc2)
        self.init_layer(self.fc3)
        self.init_layer(self.fc4)

    def forward(self, input):
        batch_size, n_concat, freq_bins = input.shape
        x = input.view(batch_size, n_concat * freq_bins)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc4(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_concat': 4, 'freq_bins': 4}]
