import torch
from collections import OrderedDict
import torch.nn as nn


class SequenceClassifier(nn.Module):
    """
        Given a sequence of image vectors, intelligently weight the importance of each member
        of the sequence and use it to predict presence/absence of a class.
    """

    def __init__(self, seq_len, in_dim, classes):
        super(SequenceClassifier, self).__init__()
        selector_operations = OrderedDict({'linear1': nn.Linear(in_dim,
            in_dim, seq_len), 'relu1': nn.ReLU(inplace=True), 'linear3': nn
            .Linear(in_dim, 1, seq_len), 'sigmoid': nn.Sigmoid()})
        self.selector = nn.Sequential(selector_operations)
        predictor_operations = OrderedDict({'linear1': nn.Linear(in_dim,
            in_dim), 'relu1': nn.ReLU(inplace=True), 'linear3': nn.Linear(
            in_dim, classes), 'sigmoid': nn.Sigmoid()})
        self.predictor = nn.Sequential(predictor_operations)

    def forward(self, X):
        selector_vector = self.selector(X)
        selected = X * selector_vector
        selected = selected.mean(axis=0)
        decision = self.predictor(selected)
        return decision


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'seq_len': 4, 'in_dim': 4, 'classes': 4}]
