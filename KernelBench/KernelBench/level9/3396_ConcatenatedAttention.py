import torch
import torch.optim
import torch.utils.data
from torch import nn


class ConcatenatedAttention(nn.Module):
    """
    ConcatenatedAttention module which uses concatenation of encoder and decoder
    attention vectors instead of summing them up
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        @param encoder_dim: feature size of encoded images
        @param decoder_dim: size of decoder's RNN
        @param attention_dim: size of the attention network
        """
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim * 2, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        att2_expanded = att2.expand_as(att1)
        att = self.full_att(self.relu(torch.cat([att1, att2_expanded], dim=2))
            ).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(
            dim=1)
        return attention_weighted_encoding, alpha


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'encoder_dim': 4, 'decoder_dim': 4, 'attention_dim': 4}]
