import torch
import torch.nn as nn


class MLPClassifier(nn.Module):

    def __init__(self, embedding_dim, label_size, hidden_dim):
        super(MLPClassifier, self).__init__()
        self.layer1 = torch.nn.Linear(embedding_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(hidden_dim, label_size)

    def forward(self, x):
        hidden = self.layer1(x)
        activation = self.relu(hidden)
        label_out = self.layer2(activation)
        label_scores = torch.log_softmax(label_out, dim=0)
        return label_scores


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'embedding_dim': 4, 'label_size': 4, 'hidden_dim': 4}]
