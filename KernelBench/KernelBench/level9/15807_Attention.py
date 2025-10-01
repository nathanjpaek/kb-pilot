import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.linear_in = nn.Linear(hidden_size, hidden_size, bias=False)

    def score(self, hidden_state, encoder_states):
        """
        Args:
            hidden_state (tgt_len(=1), batch, hidden_size)
            encoder_states (src_len, batch, hidden_size)
        Return:
            score (batch, tgt_len(=1), src_len)
        """
        h_t = hidden_state.transpose(0, 1).contiguous()
        h_s = encoder_states.transpose(0, 1).contiguous()
        _src_batch, _src_len, _src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
        h_t_ = self.linear_in(h_t_)
        h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
        h_s_ = h_s.transpose(1, 2)
        score = torch.bmm(h_t, h_s_)
        return score

    def forward(self, hidden, encoder_outputs, src_mask, tgt_mask=None):
        """
        Args:
            hidden (tgt_len(=1), batch, hidden_size)
            encoder_outputs (src_len, batch, hidden_size)
            src_mask (batch, src_len)
            tgt_mask (batch, tgt_len)
        Return:
            attn: (batch, tgt_len(=1), src_len)
        """
        tgt_len, b_size, _hiddim = hidden.size()
        src_len = encoder_outputs.size(0)
        attn_scores = self.score(hidden, encoder_outputs)
        src_mask = src_mask.unsqueeze(1).expand(b_size, tgt_len, src_len)
        attn_scores = attn_scores.masked_fill(src_mask == 0, -10000000000.0)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.unsqueeze(2).expand(b_size, tgt_len, src_len)
            attn_scores = attn_scores.masked_fill(tgt_mask == 0, -10000000000.0
                )
        return F.softmax(attn_scores, dim=2)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
