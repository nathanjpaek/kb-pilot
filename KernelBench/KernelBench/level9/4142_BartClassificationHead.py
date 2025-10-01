import torch
from torch import nn
import torch.utils.checkpoint


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim: 'int', inner_dim: 'int', pooler_dropout:
        'float'):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states: 'torch.Tensor', mask: 'torch.Tensor'):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        sent_scores = self.sigmoid(hidden_states)
        sent_scores = sent_scores.squeeze(-1) * mask.float()
        return sent_scores


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'inner_dim': 4, 'pooler_dropout': 0.5}]
