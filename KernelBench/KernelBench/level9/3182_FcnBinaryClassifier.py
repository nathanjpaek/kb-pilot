import torch
import torch.nn.functional as F
import torch.nn as nn


class FcnBinaryClassifier(nn.Module):
    """
    A fully-connected neural network with a single hidden layer and batchnorm for binary classification.

    Architecture:
        Linear(input_size, hidden_size)
        ReLU()
        BatchNorm()
        Dropout()
        Linear(hidden_size, 1)

    Args:
        input_size: size of the input vector
        hidden_size: size of the hidden layer
        dropout_prob: dropout parameter
        use_batch_norm: if True, add BatchNorm between layers
    """

    def __init__(self, input_size, hidden_size, dropout_prob=0.5,
        use_batch_norm=False):
        super().__init__()
        super(FcnBinaryClassifier, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(input_size
            ) if use_batch_norm else None
        self.dropout = nn.Dropout(p=dropout_prob)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Args:
            x: torch.FloatTensor[batch_size, input_size]

        Returns:
            torch.FloatTensor[batch_size,] probabilities of a positive class for each example in the batch
        """
        x = self.input_layer(x)
        x = F.relu(x)
        if self.batch_norm:
            x = self.dropout(x)
        x = self.output_layer(x)
        prob = F.sigmoid(x)
        return prob


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
