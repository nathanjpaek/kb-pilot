import torch
import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, z_dim, hidden_dim, class_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, class_dim)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        hidden = self.softplus(self.fc1(z))
        scores = self.softmax(self.fc2(hidden))
        return scores


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'z_dim': 4, 'hidden_dim': 4, 'class_dim': 4}]
