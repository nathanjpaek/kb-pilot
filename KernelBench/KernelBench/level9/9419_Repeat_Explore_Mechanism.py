import torch
import torch.nn as nn


class Repeat_Explore_Mechanism(nn.Module):

    def __init__(self, device, hidden_size, seq_len, dropout_prob):
        super(Repeat_Explore_Mechanism, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.device = device
        self.seq_len = seq_len
        self.Wre = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ure = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.Vre = nn.Linear(hidden_size, 1, bias=False)
        self.Wcre = nn.Linear(hidden_size, 2, bias=False)

    def forward(self, all_memory, last_memory):
        """
        calculate the probability of Repeat and explore
        """
        all_memory_values = all_memory
        all_memory = self.dropout(self.Ure(all_memory))
        last_memory = self.dropout(self.Wre(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)
        output_ere = self.tanh(all_memory + last_memory)
        output_ere = self.Vre(output_ere)
        alpha_are = nn.Softmax(dim=1)(output_ere)
        alpha_are = alpha_are.repeat(1, 1, self.hidden_size)
        output_cre = alpha_are * all_memory_values
        output_cre = output_cre.sum(dim=1)
        output_cre = self.Wcre(output_cre)
        repeat_explore_mechanism = nn.Softmax(dim=-1)(output_cre)
        return repeat_explore_mechanism


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'device': 0, 'hidden_size': 4, 'seq_len': 4,
        'dropout_prob': 0.5}]
