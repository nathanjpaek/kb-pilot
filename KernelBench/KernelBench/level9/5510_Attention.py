from torch.nn import Module
import torch
from torch.nn.modules import Module
from torch.nn.functional import softmax
from torch.nn import Linear


def neginf(dtype):
    """
    Return a representable finite 
    number near -inf for a dtype.
    """
    if dtype is torch.float16:
        return -65504
    else:
        return -1e+20


class Attention(Module):
    """
    Luong style general attention from 
    https://arxiv.org/pdf/1508.04025.pdf.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.project = Linear(in_features=hidden_size, out_features=
            hidden_size, bias=False)
        self.combine = Linear(in_features=hidden_size * 2, out_features=
            hidden_size, bias=False)

    def forward(self, decoder_output, hidden_state, encoder_outputs,
        attn_mask=None):
        """
        Applies attention by creating the weighted 
        context vector. Implementation is based on 
        `IBM/pytorch-seq2seq`.
        """
        hidden_state = self.project(hidden_state)
        hidden_state = hidden_state.transpose(0, 1)
        encoder_outputs_t = encoder_outputs.transpose(1, 2)
        attn_scores = torch.bmm(hidden_state, encoder_outputs_t)
        if attn_mask is not None:
            attn_scores = attn_scores.squeeze(1)
            attn_scores.masked_fill_(attn_mask, neginf(attn_scores.dtype))
            attn_scores = attn_scores.unsqueeze(1)
        attn_weights = softmax(attn_scores, dim=-1)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        stacked = torch.cat([decoder_output, attn_applied], dim=-1)
        outputs = self.combine(stacked)
        return outputs, attn_weights


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
