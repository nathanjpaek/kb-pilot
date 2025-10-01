import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(Attention, self).__init__()
        self.v = nn.Parameter(torch.zeros((1, 1, decoder_hidden_size),
            requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, decoder_hidden_size, 
            encoder_hidden_size + decoder_hidden_size), requires_grad=True))

    def forward(self, encoder_hidden, decoder_hidden):
        batch_size, hidden_size = decoder_hidden.size()
        decoder_hidden = decoder_hidden.unsqueeze(2).repeat(1, 1,
            encoder_hidden.shape[-1])
        hidden = torch.cat((encoder_hidden, decoder_hidden), 1)
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)
        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)
        return attns


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'encoder_hidden_size': 4, 'decoder_hidden_size': 4}]
