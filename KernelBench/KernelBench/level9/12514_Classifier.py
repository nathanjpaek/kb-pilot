import torch
import torch.nn.functional as F
from torch import nn


class Classifier(nn.Module):
    """
    Inherits Class information from the nn.Module and creates a Classifier Class:
        - Class has these attributes:
            o fully connected layer with specified number of in_features and out_features
            o number of hidden layers equivalent to the inputted requirements
            o dropout parameter for the fully connected layers
        - Class has a forward method:
            o Flattens the input data in an input layer for computation
            o Connects each layer with a relu activation, the defined dropout, and linear regression
            o Returns outputs from the final hidden layer into an categorical output probability using log_softmax
    Parameters:
        - in_features
        - hidden_layers
        - out_features
    """

    def __init__(self, in_features, hidden_layers, out_features):
        super().__init__()
        self.in_features = in_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self._index = 1
        while self._index < self.hidden_layers:
            setattr(self, 'fc' + str(self._index), nn.Linear(round(self.
                in_features / 2 ** (self._index - 1)), round(self.
                in_features / 2 ** self._index)))
            self._index += 1
        setattr(self, 'fc' + str(self._index), nn.Linear(round(self.
            in_features / 2 ** (self._index - 1)), self.out_features))
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        self._index = 1
        while self._index < self.hidden_layers:
            x = self.dropout(F.relu(getattr(self, 'fc' + str(self._index))(x)))
            self._index += 1
        x = F.log_softmax(getattr(self, 'fc' + str(self._index))(x), dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'hidden_layers': 1, 'out_features': 4}]
