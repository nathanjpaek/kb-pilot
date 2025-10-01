import math
import torch
import torch.nn.functional as F


class MultiHeadedAttention(torch.nn.Module):
    """
    Implement of multi-head attention.
    """

    def __init__(self, n_heads, hidden_size, drop_rate):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_dk = hidden_size // n_heads
        self.n_heads = n_heads
        self.proj_query = torch.nn.Linear(hidden_size, hidden_size)
        self.proj_key = torch.nn.Linear(hidden_size, hidden_size)
        self.proj_value = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(drop_rate)
        self.proj_output = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, input_, mask=None):
        """
        Input: embedding.
        """
        batch_size = input_.size(0)
        query = self.proj_query(input_)
        query = query.view(batch_size, -1, self.n_heads, self.n_dk).transpose(
            1, 2)
        key = self.proj_key(input_)
        key = key.view(batch_size, -1, self.n_heads, self.n_dk).transpose(1, 2)
        value = self.proj_value(input_)
        value = value.view(batch_size, -1, self.n_heads, self.n_dk).transpose(
            1, 2)
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.n_dk)
        if mask is not None:
            mask = mask[:, None, None, :]
            scores = scores.masked_fill(mask == 0, -1000000000.0)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        cv = attn @ value
        cv = cv.transpose(1, 2)
        cv = cv.contiguous().view(batch_size, -1, self.n_heads * self.n_dk)
        return self.proj_output(cv)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_heads': 4, 'hidden_size': 4, 'drop_rate': 0.5}]
