import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.onnx.operators


def get_activation_fn(activation):
    """
    Get activation function by name

    Args:
        activation: activation function name

    Returns:
        - activation function
    """
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    else:
        raise KeyError


class HuggingfaceClassifier(nn.Module):
    """
    Classifier implemented in HuggingfaceClassificationHead style.

    Args:
        d_model: feature dimensionality
        labels: number of classes
        inner_dim: dimensionality in the inner vector space.
        activation: activation function used in the feed-forward network
        dropout: dropout rate
    """

    def __init__(self, d_model, labels, inner_dim=None, activation='relu',
        dropout=0.0):
        super().__init__()
        inner_dim = inner_dim or d_model * 2
        self._fc1 = nn.Linear(d_model, inner_dim)
        self._dropout = nn.Dropout(dropout)
        self._fc2 = nn.Linear(inner_dim, labels)
        self._activation = get_activation_fn(activation)

    def forward(self, x):
        """
        Args:
            x: feature to predict labels
                :math:`(*, D)`, where D is the feature dimension

        Returns:
            - log probability of each classes
                :math: `(*, L)`, where L is the number of classes
        """
        x = self._dropout(x)
        x = self._fc1(x)
        x = self._activation(x)
        x = self._dropout(x)
        x = self._fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'labels': 4}]
