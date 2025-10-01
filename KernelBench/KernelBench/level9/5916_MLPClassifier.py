import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """MLP Classifier."""

    def __init__(self, input_dim: 'int', hidden_dim: 'int', output_dim:
        'int', dropout: 'float'=0.0, nonlinearity: 'str'='tanh',
        batch_first: 'bool'=True, **kwargs) ->None:
        """
        Initialise the model.

        :input_dim (int): The dimension of the input to the model.
        :hidden_dim (int): The dimension of the hidden layer.
        :output_dim (int): The dimension of the output layer (i.e. the number of classes).
        :dropout (float, default = 0.0): Value of dropout layer.
        :nonlinearity (str, default = 'tanh'): String name of nonlinearity function to be used.
        :batch_first (bool): Batch the first dimension?
        """
        super(MLPClassifier, self).__init__()
        self.batch_first = batch_first
        self.name = 'onehot_mlp'
        self.info = {'Model': self.name, 'Input dim': input_dim,
            'Hidden dim': hidden_dim, 'Output dim': output_dim,
            'nonlinearity': nonlinearity, 'Dropout': dropout}
        self.itoh = nn.Linear(input_dim, hidden_dim)
        self.htoh = nn.Linear(hidden_dim, hidden_dim)
        self.htoo = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.nonlinearity = (torch.relu if nonlinearity == 'relu' else
            torch.tanh)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sequence: 'base.DataType'):
        """
        Forward step in the classifier.

        :sequence: The sequence to pass through the network.
        :return (base.DataType): The "probability" distribution for the classes.
        """
        if self.batch_first:
            sequence = sequence.transpose(0, 1)
        sequence = sequence.float()
        out = self.dropout(self.nonlinearity(self.itoh(sequence)))
        out = self.dropout(self.nonlinearity(self.htoh(out)))
        out = out.mean(0)
        out = self.htoo(out)
        prob_dist = self.softmax(out)
        return prob_dist


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4}]
