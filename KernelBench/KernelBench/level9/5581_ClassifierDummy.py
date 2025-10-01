import torch
import torch.distributed
import torch
import torch.nn as nn


class ClassifierDummy(nn.Module):

    def __init__(self, hidden_size):
        super(ClassifierDummy, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.softmax(h) * mask_cls.float()
        return sent_scores


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
