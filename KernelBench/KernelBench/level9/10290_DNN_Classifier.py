import torch
from torch import nn


class DNN_Classifier(torch.nn.Module):

    def __init__(self, input_dim, nb_categories, hidden_dim=100):
        super(DNN_Classifier, self).__init__()
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, nb_categories)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_in):
        x = torch.relu(self.fc_1(x_in))
        x = self.softmax(self.fc_2(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'nb_categories': 4}]
