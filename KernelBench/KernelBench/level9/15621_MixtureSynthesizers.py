import torch
import torch.nn as nn


class MixtureSynthesizers(nn.Module):

    def __init__(self, in_dims, sentence_length):
        super(MixtureSynthesizers, self).__init__()
        self.attention = nn.Parameter(torch.empty(1, sentence_length,
            sentence_length), requires_grad=True)
        nn.init.xavier_uniform_(self.attention)
        self.query_fc = nn.Linear(in_dims, in_dims)
        self.key_fc = nn.Linear(in_dims, in_dims)
        self.value_fc = nn.Linear(in_dims, in_dims)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query_fc(x)
        key = self.key_fc(x).permute(0, 2, 1)
        vanilla_energy = torch.bmm(query, key)
        energy = self.attention + vanilla_energy
        attention = self.softmax(energy)
        value = self.value_fc(x)
        out = torch.bmm(attention, value)
        return out, attention


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dims': 4, 'sentence_length': 4}]
