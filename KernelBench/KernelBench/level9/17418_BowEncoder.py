import torch
import torch.nn as nn
from torch.nn import functional as F


class BowEncoder(nn.Module):
    """
    static information extractor
    """

    def __init__(self, num_words, bow_mid_hid, dropout):
        super().__init__()
        self.fc1 = nn.Linear(num_words, bow_mid_hid)
        self.fc_trans = nn.Linear(bow_mid_hid, bow_mid_hid)
        self.dropout = nn.Dropout(dropout)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc_trans.weight)

    def forward(self, x_bow):
        x_bow = F.relu(self.fc1(x_bow))
        x_bow = F.relu(self.fc_trans(x_bow))
        return x_bow


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_words': 4, 'bow_mid_hid': 4, 'dropout': 0.5}]
