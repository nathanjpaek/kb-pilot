import torch
import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = x.squeeze()
        out = self.fc1(x)
        out = self.softmax(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4, 'num_classes': 4}]
