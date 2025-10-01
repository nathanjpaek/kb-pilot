import torch
import torch.nn as nn


class criticAttention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(criticAttention, self).__init__()
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
            requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
            requires_grad=True))

    def forward(self, encoder_hidden, decoder_hidden):
        batch_size, hidden_size, _ = encoder_hidden.size()
        hidden = decoder_hidden.unsqueeze(2).expand_as(encoder_hidden)
        hidden = torch.cat((encoder_hidden, hidden), 1)
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)
        logit = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        logit = torch.softmax(logit, dim=2)
        return logit


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
