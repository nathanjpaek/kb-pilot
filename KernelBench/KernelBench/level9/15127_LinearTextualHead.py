import torch
import torch.nn as nn
from typing import Optional


class TextualHead(nn.Module):
    """
    Base class for all textual heads. All child classes can simply inherit
    from :class:`~torch.nn.Module`, however this is kept here for uniform
    type annotations.

    Parameters
    ----------
    visual_feature_size: int
        Size (number of channels) of the input features from the visual backbone.
    vocab_size: int
        Number of tokens in the output vocabulary.
    hidden_size: int
        Size of the token embedding vectors, or hidden state vector of the
        language model.
    """

    def __init__(self, visual_feature_size: 'int', vocab_size: 'int',
        hidden_size: 'int'):
        super().__init__()
        self.visual_feature_size = visual_feature_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    @property
    def textual_feature_size(self):
        """
        Size of the last dimension of output right before the output linear
        layer (which predicts a distribution over vocabulary tokens). This is
        typically same as :attr:`hidden_size` for most modules. This property
        is used to add more modules on top of this.
        """
        return self.hidden_size


class LinearTextualHead(TextualHead):
    """
    A textual head containing a single linear layer projecting from the visual
    feature size to the output vocabulary size.

    Parameters
    ----------
    visual_feature_size: int
        Size (number of channels) of the input features from the visual backbone.
    vocab_size: int
        Number of tokens in the output vocabulary.
    """

    def __init__(self, visual_feature_size: 'int', vocab_size: 'int', **kwargs
        ):
        hidden_size = visual_feature_size
        super().__init__(visual_feature_size, vocab_size, hidden_size)
        self.output = nn.Linear(visual_feature_size, vocab_size)

    def forward(self, visual_features: 'torch.Tensor', caption_tokens:
        'Optional[torch.Tensor]'=None, caption_lengths:
        'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """
        Project visual features directly to predict a distribution over
        vocabulary tokens through a single linear layer. This textual head
        ignores arguments ``caption_tokens`` and ``caption_lengths``, they
        are here for API consistency.

        Parameters
        ----------
        visual_features: torch.Tensor
            A tensor of shape ``(batch_size, channels, height, width)`` containing
            features from visual backbone.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size, vocab_size)`` containing output
            vocabulary logits.
        """
        batch_size, channels, _height, _width = visual_features.size()
        visual_features = visual_features.view(batch_size, channels, -1)
        visual_features = visual_features.permute(0, 2, 1)
        visual_features = visual_features.mean(dim=1)
        output_logits = self.output(visual_features)
        return output_logits


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'visual_feature_size': 4, 'vocab_size': 4}]
