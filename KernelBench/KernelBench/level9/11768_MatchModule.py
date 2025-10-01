import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchModule(nn.Module):
    """
    Computing the match representation for Match LSTM.

    :param hidden_size: Size of hidden vectors.
    :param dropout_rate: Dropout rate of the projection layer. Defaults to 0.

    Examples:
        >>> import torch
        >>> attention = MatchModule(hidden_size=10)
        >>> v1 = torch.randn(4, 5, 10)
        >>> v1.shape
        torch.Size([4, 5, 10])
        >>> v2 = torch.randn(4, 5, 10)
        >>> v2_mask = torch.ones(4, 5).to(dtype=torch.uint8)
        >>> attention(v1, v2, v2_mask).shape
        torch.Size([4, 5, 20])


    """

    def __init__(self, hidden_size, dropout_rate=0):
        """Init."""
        super().__init__()
        self.v2_proj = nn.Linear(hidden_size, hidden_size)
        self.proj = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, v1, v2, v2_mask):
        """Computing attention vectors and projection vectors."""
        proj_v2 = self.v2_proj(v2)
        similarity_matrix = v1.bmm(proj_v2.transpose(2, 1).contiguous())
        v1_v2_attn = F.softmax(similarity_matrix.masked_fill(v2_mask.
            unsqueeze(1).bool(), -1e-07), dim=2)
        v2_wsum = v1_v2_attn.bmm(v2)
        fusion = torch.cat([v1, v2_wsum, v1 - v2_wsum, v1 * v2_wsum], dim=2)
        match = self.dropout(F.relu(self.proj(fusion)))
        return match


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
