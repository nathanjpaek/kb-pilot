import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def softmax_mask(self, val, mask):
        rank_val = len(list(val.shape))
        rank_mask = len(list(mask.shape))
        if rank_val - rank_mask == 1:
            mask = torch.unsqueeze(mask, axis=-1)
        return (0 - 1e+30) * (1 - mask.float()) + val

    def forward(self, inputs, mask=None, keep_prob=1.0, is_train=True):
        x = torch.dropout(inputs, keep_prob, is_train)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        if mask is not None:
            x = self.softmax_mask(x, mask)
        x = F.softmax(x, dim=1)
        x = x.squeeze(-1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
