import torch
import torch.nn as nn


class Classifier(nn.Module):
    """MLP classifier

    Parameters
    ----------
    n_dimensions : int
        Embedding dimension
    n_classes : int
        Number of classes.
    """

    def __init__(self, n_dimensions, n_classes):
        super().__init__()
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.hidden = nn.Linear(n_dimensions, n_dimensions, bias=True)
        self.output = nn.Linear(n_dimensions, n_classes, bias=True)
        self.logsoftmax_ = nn.LogSoftmax(dim=-1)

    def forward(self, embedding):
        hidden = torch.tanh(self.hidden(embedding))
        return self.logsoftmax_(self.output(hidden))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_dimensions': 4, 'n_classes': 4}]
