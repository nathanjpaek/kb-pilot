import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    Implements Bahdanau Attention.

    Arguments:
        encoder_dim (int):  Size of the encoder.
        decoder_dim (int):  Size of the decoder.
        attention_dim (int):  Size of the attention layer.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_attn = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_attn = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.value = nn.Linear(attention_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        hidden_ = decoder_hidden.unsqueeze(1)
        attn_hidden = torch.tanh(self.encoder_attn(encoder_out) + self.
            decoder_attn(hidden_))
        score = self.value(attn_hidden)
        attn_weights = self.softmax(score)
        context_vector = attn_weights * encoder_out
        context_vector = torch.sum(context_vector, 1)
        return context_vector, attn_weights


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'encoder_dim': 4, 'decoder_dim': 4, 'attention_dim': 4}]
