import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """
    Attention mechansims usually scale values based on relationships between
    keys and queries. 
    
    Attention(Q,K,V) = A(Q,K)*V where A() is a normalization function.

    A common choice for the normalization function is scaled dot-product attention:

    A(Q,K) = Softmax(Q*K^T / sqrt(d_attention))

    Args:
          dropout (float): Fraction between 0 and 1 corresponding to the degree of dropout used
    """

    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value, mask=None):
        """
        Args:
          query (torch.tensor): 
          key (torch.tensor):
          value (torch.tensor): 
          mask (torch.tensor):
        """
        d_k = key.shape[-1]
        scaling_factor = torch.sqrt(torch.tensor(d_k))
        scaled_dot_product = torch.matmul(query, key.permute(0, 2, 1)
            ) / scaling_factor
        if mask is not None:
            scaled_dot_product = scaled_dot_product.masked_fill(mask == 0, 
                -1000000000.0)
        attention = self.softmax(scaled_dot_product)
        attention = self.dropout(attention)
        output = torch.matmul(attention, value)
        return output, attention


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {}]
