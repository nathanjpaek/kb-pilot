import torch
import torch.nn as nn


class MessageNormalizer(nn.Module):

    def __init__(self, in_features, init_mean=1.0, init_stddev=0.01):
        super(MessageNormalizer, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features))
        self.init_mean = init_mean
        self.init_stddev = init_stddev
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(mean=self.init_mean, std=self.init_stddev)

    def forward(self, message):
        return self.weight * message

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.
            in_features) + 'out_features=' + str(self.out_features) + ')'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4}]
