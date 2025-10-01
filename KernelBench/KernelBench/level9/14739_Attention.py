import torch
from torch import nn
import torch.nn.utils


class Attention(nn.Module):

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.ff = nn.Linear(in_features=hidden_dim, out_features=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, contexts, context_masks=None):
        """
        :param contexts: (batch_size, seq_len, n_hid)
        :param context_masks: (batch_size, seq_len)
        :return: (batch_size, n_hid), (batch_size, seq_len)
        """
        out = self.ff(contexts)
        out = out.view(contexts.size(0), contexts.size(1))
        if context_masks is not None:
            masked_out = out.masked_fill(context_masks, float('-inf'))
        else:
            masked_out = out
        attn_weights = self.softmax(masked_out)
        out = attn_weights.unsqueeze(1).bmm(contexts)
        out = out.squeeze(1)
        return out, attn_weights


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4}]
