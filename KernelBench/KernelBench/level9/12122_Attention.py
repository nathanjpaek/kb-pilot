import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, device, hidden_size):
        super(Attention, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.concat_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, hidden_size)

    def forward(self, rnn_outputs, final_hidden_state):
        _batch_size, _seq_len, _ = rnn_outputs.shape
        attn_weights = self.attn(rnn_outputs)
        attn_weights = torch.bmm(attn_weights, final_hidden_state.unsqueeze(2))
        attn_weights = F.softmax(attn_weights.squeeze(2), dim=1)
        context = torch.bmm(rnn_outputs.transpose(1, 2), attn_weights.
            unsqueeze(2)).squeeze(2)
        attn_hidden = torch.tanh(self.concat_linear(torch.cat((context,
            final_hidden_state), dim=1)))
        return attn_hidden, attn_weights


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'device': 0, 'hidden_size': 4}]
