import torch
import torch.nn as nn
import torch.nn.functional as F


class Classification(torch.nn.Module):

    def __init__(self, num_class, hidden_dim):
        super(Classification, self).__init__()
        self.num_class = num_class
        self.label = nn.Linear(hidden_dim, num_class)

    def forward(self, input):
        outp = self.label(input)
        class_score = F.log_softmax(outp.view(-1, self.num_class), dim=1)
        return class_score


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_class': 4, 'hidden_dim': 4}]
