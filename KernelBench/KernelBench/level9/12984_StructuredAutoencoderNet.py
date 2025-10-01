from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
from collections import OrderedDict


class StructuredAutoencoderNet(nn.Module):

    def __init__(self, p, encoder_config, decoder_config, dropout_rate=0):
        super().__init__()
        self.p = p
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.weights_layer = []
        index = 0
        self.encoder_layer = []
        for i in range(len(self.encoder_config['dimension']) - 1):
            self.encoder_layer.append(('linear' + str(index), nn.Linear(int
                (self.encoder_config['dimension'][i]), int(self.
                encoder_config['dimension'][i + 1]))))
            if i != len(self.encoder_config['dimension']) - 2:
                self.encoder_layer.append(('Sigmoid' + str(index), nn.
                    Sigmoid()))
                self.encoder_layer.append(('dropout' + str(index), nn.
                    Dropout(p=dropout_rate)))
            index += 1
        for index, layer in enumerate(self.encoder_layer):
            if layer[0] == 'linear':
                self.weights_layer.append(torch.nn.Parameter(layer[1].weight))
                self.encoder_layer[index][1].weight = self.weights_layer[-1]
        index = 0
        self.decoder_layer = []
        for i in range(len(self.decoder_config['dimension']) - 1):
            if i != 0:
                self.decoder_layer.append(('dropout' + str(index), nn.
                    Dropout(p=dropout_rate)))
            self.decoder_layer.append(('linear' + str(index), nn.Linear(int
                (self.decoder_config['dimension'][i]), int(self.
                decoder_config['dimension'][i + 1]))))
            if i != len(self.decoder_config['dimension']) - 2:
                self.decoder_layer.append(('Sigmoid' + str(index), nn.
                    Sigmoid()))
            index += 1
        self.encoder_net = nn.Sequential(OrderedDict(self.encoder_layer))
        self.decoder_net = nn.Sequential(OrderedDict(self.decoder_layer))

    def encode(self, X, mask):
        index = 0
        for layer in self.encoder_layer:
            if layer[0] == 'linear':
                X = torch.nn.functional.linear(X, self.weights_layer[index])
                index += 1
            else:
                X = layer[1](X)
        X = X * mask
        return X

    def decode(self, X):
        index = len(self.weights_layer) - 1
        for layer in self.decoder_layer:
            if layer[0] == 'linear':
                X = torch.nn.functional.linear(X, self.weights_layer[index].t()
                    )
                index -= 1
            else:
                X = layer[1](X)
        return X

    def forward(self, X, mask):
        X = self.encode(X, mask)
        X = self.decode(X)
        return X


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'p': 4, 'encoder_config': _mock_config(dimension=[4, 4]),
        'decoder_config': _mock_config(dimension=[4, 4])}]
