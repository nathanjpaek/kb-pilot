import torch
import torch.nn as nn
import torch.nn.functional as F


class Luong_Attention(nn.Module):

    def __init__(self, hidden_size, score='general'):
        super(Luong_Attention, self).__init__()
        assert score.lower() in ['concat', 'general', 'dot']
        self.score = score.lower()

        def wn(x):
            return nn.utils.weight_norm(x)
        if self.score == 'general':
            self.attn = wn(nn.Linear(hidden_size, hidden_size))
        elif self.score == 'concat':
            raise Exception('concat disabled for now. results are poor')
            self.attn = wn(nn.Linear(2 * hidden_size, hidden_size))
            self.v = wn(nn.Linear(hidden_size, 1))

    def forward(self, hidden_state, encoder_outputs):
        assert hidden_state.size(1) == encoder_outputs.size(1)
        assert len(hidden_state.size()) == 3
        hidden_state = hidden_state.transpose(1, 0).contiguous()
        encoder_outputs = encoder_outputs.transpose(1, 0).contiguous()
        if self.score == 'dot':
            grid = torch.bmm(hidden_state, encoder_outputs.transpose(2, 1))
        elif self.score == 'general':
            grid = torch.bmm(hidden_state, self.attn(encoder_outputs).
                transpose(2, 1))
        elif self.score == 'concat':
            cc = self.attn(torch.cat((hidden_state.expand(encoder_outputs.
                size()), encoder_outputs), 2))
            grid = self.v(cc)
            grid = grid.permute(0, 2, 1)
        mask = (grid != 0).float()
        attn_weights = F.softmax(grid, dim=2) * mask
        normalizer = attn_weights.sum(dim=2).unsqueeze(2)
        attn_weights /= normalizer
        return attn_weights


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
