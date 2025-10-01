import torch
from torch import nn
import torch.optim
import torch.utils.data


class PositionWiseFCNetwork(nn.Module):
    """
    The Position-Wise Feed Forward Network sublayer.
    """

    def __init__(self, d_model, d_inner, dropout):
        """
        :param d_model: size of vectors throughout the transformer model, i.e. input and output sizes for this sublayer
        :param d_inner: an intermediate size
        :param dropout: dropout probability
        """
        super(PositionWiseFCNetwork, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_inner)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_inner, d_model)
        self.apply_dropout = nn.Dropout(dropout)

    def forward(self, sequences):
        """
        Forward prop.

        :param sequences: input sequences, a tensor of size (N, pad_length, d_model)
        :return: transformed output sequences, a tensor of size (N, pad_length, d_model)
        """
        input_to_add = sequences.clone()
        sequences = self.layer_norm(sequences)
        sequences = self.apply_dropout(self.relu(self.fc1(sequences)))
        sequences = self.fc2(sequences)
        sequences = self.apply_dropout(sequences) + input_to_add
        return sequences


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_inner': 4, 'dropout': 0.5}]
