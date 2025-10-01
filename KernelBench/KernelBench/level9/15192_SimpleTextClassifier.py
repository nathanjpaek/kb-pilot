import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTextClassifier(nn.Module):
    """Text Classifier with 1 hidden layer 

    """

    def __init__(self, num_labels, vocab_size):
        super(SimpleTextClassifier, self).__init__()
        self.linear1 = nn.Linear(vocab_size, 128)
        self.linear2 = nn.Linear(128, num_labels)

    def forward(self, feature_vec):
        hidden1 = self.linear1(feature_vec).clamp(min=0)
        output = self.linear2(hidden1)
        return F.log_softmax(output, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_labels': 4, 'vocab_size': 4}]
