import torch
from torch import Tensor
from typing import Tuple
import torch.nn as nn


class SLMLexicon(nn.Module):
    """
    The optional "Lexicon" or "Memory" component of the Segmental Language
    Model. Decodes context/position encodings to logits over a segmental
    vocabulary, as well as a mixture proportion for combining this loss with the
    character-generation loss

    Args:
        d_enc: The dimension of the encodings returned from the encoder
        d_model: The dimension of the hidden states used in the decoder and the
            rest of the model
        subword_vocab_size: The size of the vocabulary over subwords/segments
        initrange: The positive end of the initialization range for the lexicon
            layers. Default: 0.1
    """

    def __init__(self, d_enc: 'int', d_model: 'int', subword_vocab_size:
        'int', initrange: 'float'=0.1):
        super().__init__()
        self.encoding_to_subword_hidden = nn.Linear(d_enc, d_model)
        self.subword_decoder = nn.Linear(d_model, subword_vocab_size)
        self.encoding_to_mixture_hidden = nn.Linear(d_enc, d_model)
        self.hidden_to_mixture_proportion = nn.Linear(d_model, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.encoding_to_subword_hidden.weight.data.uniform_(-initrange,
            initrange)
        self.subword_decoder.weight.data.uniform_(-initrange, initrange)
        self.encoding_to_mixture_hidden.weight.data.uniform_(-initrange,
            initrange)
        self.hidden_to_mixture_proportion.weight.data.uniform_(-initrange,
            initrange)
        self.encoding_to_subword_hidden.bias.data.zero_()
        self.subword_decoder.bias.data.zero_()
        self.encoding_to_mixture_hidden.bias.data.zero_()

    def forward(self, encodings: 'Tensor') ->Tuple[Tensor, Tensor]:
        """
        Decode the segment encodings to logits over the subword vocabulary and
        mixture proportions for the Lexicon

        Args:
            encodings: The context/positional encodings output by the SLM
                Encoder
        """
        subword_encodings = self.encoding_to_subword_hidden(encodings)
        subword_scores = self.subword_decoder(subword_encodings)
        subword_probs = self.log_softmax(subword_scores)
        mixture_encodings = self.encoding_to_mixture_hidden(encodings)
        mixture_outputs = self.hidden_to_mixture_proportion(mixture_encodings)
        mixture_proportions = self.sigmoid(mixture_outputs.squeeze(-1))
        return subword_probs, mixture_proportions


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_enc': 4, 'd_model': 4, 'subword_vocab_size': 4}]
