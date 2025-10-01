import torch
import torch.nn as nn


class StackedAutoencoder(nn.Module):
    """
    1-hidden layer AE trained with MSE loss
    """

    def __init__(self, input_size, hidden_layer_size):
        super(StackedAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_layer_size)
        self.decoder = nn.Linear(hidden_layer_size, input_size)

    def embedding(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.sigmoid(x)
        return self.decoder(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_layer_size': 1}]
