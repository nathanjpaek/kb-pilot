import math
import torch
import torch.nn.functional as F


def gelu(x):
    """
    GELU activation function.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


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


class LayerNormalization(torch.nn.Module):
    """
    Epsilon outsize the square root.
    """

    def __init__(self, size, eps=1e-06):
        super(LayerNormalization, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(size))
        self.beta = torch.nn.Parameter(torch.zeros(size))
        self.eps = eps
        self.register_parameter('gamma', self.gamma)
        self.register_parameter('beta', self.beta)

    def forward(self, input_):
        mean = torch.mean(input_, -1, keepdim=True)
        std = torch.std(input_, -1, keepdim=True)
        return self.gamma * (input_ - mean) / (std + self.eps) + self.beta


class PositionwiseFeedForward(torch.nn.Module):
    """
    FeedForward Neural Networks for each position
    """

    def __init__(self, input_size, hidden_size, output_size, drop_rate):
        super(PositionwiseFeedForward, self).__init__()
        self.ff1 = torch.nn.Linear(input_size, hidden_size)
        self.ff2 = torch.nn.Linear(hidden_size, output_size)
        self.drop = torch.nn.Dropout(drop_rate)

    def forward(self, input_):
        """ 
        (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        """
        return self.drop(self.ff2(gelu(self.ff1(input_))))


class TransformerBlock(torch.nn.Module):
    """
    Implementation of Transformer
    """

    def __init__(self, input_size, n_heads, drop_rate, device=torch.device(
        'cpu')):
        super().__init__()
        self.attentionMH = MultiHeadedAttention(n_heads, input_size, drop_rate)
        self.norm1 = LayerNormalization(input_size)
        self.norm2 = LayerNormalization(input_size)
        self.layer_ff = PositionwiseFeedForward(input_size, input_size * 4,
            input_size, drop_rate)
        self.drop = torch.nn.Dropout(drop_rate)

    def forward(self, input_, mask=None):
        """
        Transformer
        """
        hd = self.attentionMH(input_, mask)
        hd = self.norm1(input_ + self.drop(hd))
        hd = self.norm2(hd + self.layer_ff(hd))
        return self.drop(hd)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'n_heads': 4, 'drop_rate': 0.5}]
