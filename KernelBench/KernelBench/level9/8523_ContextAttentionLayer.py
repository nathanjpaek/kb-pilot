import torch
from collections import OrderedDict
import torch.nn as nn


class Squeeze(nn.Module):
    """Squeeze wrapper for nn.Sequential."""

    def forward(self, data):
        return torch.squeeze(data)


class Temperature(nn.Module):
    """Temperature wrapper for nn.Sequential."""

    def __init__(self, temperature):
        super(Temperature, self).__init__()
        self.temperature = temperature

    def forward(self, data):
        return data / self.temperature


class ContextAttentionLayer(nn.Module):
    """
    Implements context attention as in the PaccMann paper (Figure 2C) in
    Molecular Pharmaceutics.
    With the additional option of having a hidden size in the context.
    NOTE:
    In tensorflow, weights were initialized from N(0,0.1). Instead, pytorch
    uses U(-stddev, stddev) where stddev=1./math.sqrt(weight.size(1)).
    """

    def __init__(self, reference_hidden_size: 'int',
        reference_sequence_length: 'int', context_hidden_size: 'int',
        context_sequence_length: 'int'=1, attention_size: 'int'=16,
        individual_nonlinearity: 'type'=nn.Sequential(), temperature:
        'float'=1.0):
        """Constructor
        Arguments:
            reference_hidden_size (int): Hidden size of the reference input
                over which the attention will be computed (H).
            reference_sequence_length (int): Sequence length of the reference
                (T).
            context_hidden_size (int): This is either simply the amount of
                features used as context (G) or, if the context is a sequence
                itself, the hidden size of each time point.
            context_sequence_length (int): Hidden size in the context, useful
                if context is also textual data, i.e. coming from nn.Embedding.
                Defaults to 1.
            attention_size (int): Hyperparameter of the attention layer,
                defaults to 16.
            individual_nonlinearities (type): This is an optional
                nonlinearity applied to each projection. Defaults to
                nn.Sequential(), i.e. no nonlinearity. Otherwise it expects a
                torch.nn activation function, e.g. nn.ReLU().
            temperature (float): Temperature parameter to smooth or sharpen the
                softmax. Defaults to 1. Temperature > 1 flattens the
                distribution, temperature below 1 makes it spikier.
        """
        super().__init__()
        self.reference_sequence_length = reference_sequence_length
        self.reference_hidden_size = reference_hidden_size
        self.context_sequence_length = context_sequence_length
        self.context_hidden_size = context_hidden_size
        self.attention_size = attention_size
        self.individual_nonlinearity = individual_nonlinearity
        self.temperature = temperature
        self.reference_projection = nn.Sequential(OrderedDict([(
            'projection', nn.Linear(reference_hidden_size, attention_size)),
            ('act_fn', individual_nonlinearity)]))
        self.context_projection = nn.Sequential(OrderedDict([('projection',
            nn.Linear(context_hidden_size, attention_size)), ('act_fn',
            individual_nonlinearity)]))
        if context_sequence_length > 1:
            self.context_hidden_projection = nn.Sequential(OrderedDict([(
                'projection', nn.Linear(context_sequence_length,
                reference_sequence_length)), ('act_fn',
                individual_nonlinearity)]))
        else:
            self.context_hidden_projection = nn.Sequential()
        self.alpha_projection = nn.Sequential(OrderedDict([('projection',
            nn.Linear(attention_size, 1, bias=False)), ('squeeze', Squeeze(
            )), ('temperature', Temperature(self.temperature)), ('softmax',
            nn.Softmax(dim=1))]))

    def forward(self, reference: 'torch.Tensor', context: 'torch.Tensor'):
        """
        Forward pass through a context attention layer
        Arguments:
            reference (torch.Tensor): This is the reference input on which
                attention is computed.
                Shape: batch_size x ref_seq_length x ref_hidden_size
            context (torch.Tensor): This is the context used for attention.
                Shape: batch_size x context_seq_length x context_hidden_size
        Returns:
            (output, attention_weights):  A tuple of two Tensors, first one
                containing the reference filtered by attention (shape:
                batch_size x context_hidden_size x 1) and the second one the
                attention weights (batch_size x context_sequence_length x 1).
        """
        assert len(reference.shape) == 3, 'Reference tensor needs to be 3D'
        assert len(context.shape) == 3, 'Context tensor needs to be 3D'
        reference_attention = self.reference_projection(reference)
        context_attention = self.context_hidden_projection(self.
            context_projection(context).permute(0, 2, 1)).permute(0, 2, 1)
        alphas = self.alpha_projection(torch.tanh(reference_attention +
            context_attention))
        output = torch.sum(reference * torch.unsqueeze(alphas, -1), 1)
        return output, alphas


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'reference_hidden_size': 4, 'reference_sequence_length': 4,
        'context_hidden_size': 4}]
