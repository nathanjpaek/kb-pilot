import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):

    def __init__(self, hidden_size, batch_size=1, method='dot'):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size, bias=False)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size, bias=False
                )
            self.v = nn.Parameter(torch.FloatTensor(batch_size, 1, hidden_size)
                )

    def forward(self, hidden, encoder_outputs):
        attn_energies = self.score(hidden, encoder_outputs)
        return F.softmax(attn_energies, dim=2)

    def score(self, hidden, encoder_output):
        if self.method == 'general':
            energy = self.attn(encoder_output)
            energy = energy.transpose(2, 1)
            energy = hidden.bmm(energy)
            return energy
        elif self.method == 'concat':
            hidden = hidden * encoder_output.new_ones(encoder_output.size())
            energy = self.attn(torch.cat((hidden, encoder_output), -1))
            energy = energy.transpose(2, 1)
            energy = self.v.bmm(energy)
            return energy
        else:
            encoder_output = encoder_output.transpose(2, 1)
            energy = hidden.bmm(encoder_output)
            return energy


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
