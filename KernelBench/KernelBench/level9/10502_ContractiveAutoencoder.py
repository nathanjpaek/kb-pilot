import torch
import torch.utils.data
import torch.nn as nn


class ContractiveAutoencoder(nn.Module):
    """
    Simple contractive autoencoder with a single hidden layer.

    Constructor parameters:
        - num_inputs: Number of input features
        - num_hidden_layer_inputs: Number of input features for the single hidden layer
    """

    def __init__(self, num_inputs, num_hidden_layer_inputs):
        super(ContractiveAutoencoder, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden_layer_inputs = num_hidden_layer_inputs
        self.fc1 = nn.Linear(num_inputs, num_hidden_layer_inputs, bias=False)
        self.fc2 = nn.Linear(num_hidden_layer_inputs, num_inputs, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        h1 = self.relu(self.fc1(x.view(-1, self.num_inputs)))
        return h1

    def decoder(self, z):
        h2 = self.sigmoid(self.fc2(z))
        return h2

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_inputs': 4, 'num_hidden_layer_inputs': 1}]
