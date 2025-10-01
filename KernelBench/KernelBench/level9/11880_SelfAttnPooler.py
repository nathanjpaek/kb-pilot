import torch
import torch.nn as nn


class SelfAttnPooler(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, encoder_out, padding_mask):
        """
        encoder_out: T, B, C
        padding_mask: T, B (True for padded positions)
        """
        attn_weights = self.proj(encoder_out).squeeze(-1).float()
        if padding_mask is not None:
            attn_weights[padding_mask] = float('-inf')
        attn_weights = attn_weights.softmax(dim=0)
        out = torch.einsum('tb,tbc->bc', attn_weights.float(), encoder_out.
            float())
        return out


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.ones([4, 4, 4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'input_dim': 4}]
