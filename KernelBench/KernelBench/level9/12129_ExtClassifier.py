import torch
import torch.nn as nn
import torch.cuda
import torch.distributed


class ExtClassifier(nn.Module):

    def __init__(self, hidden_size):
        super(ExtClassifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask=None):
        h = self.linear1(x).squeeze(-1)
        if mask:
            sent_scores = self.sigmoid(h) * mask.float()
        else:
            sent_scores = self.sigmoid(h)
        return sent_scores


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
