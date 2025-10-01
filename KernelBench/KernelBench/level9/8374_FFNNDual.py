import torch
import torch.utils.data
from torch import nn


class FFNNDual(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2,
        hidden_dropout_prob_1, hidden_dropout_prob_2):
        super(FFNNDual, self).__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_dropout_prob_1 = hidden_dropout_prob_1
        self.hidden_dropout_prob_2 = hidden_dropout_prob_2
        self.dropout_1 = nn.Dropout(hidden_dropout_prob_1)
        self.dropout_2 = nn.Dropout(hidden_dropout_prob_2)
        self.first_layer = nn.Linear(input_size, hidden_size_1)
        self.second_layer = nn.Linear(hidden_size_1, hidden_size_2)
        self.classifier = nn.Linear(hidden_size_2, 2)

    def forward(self, x):
        x = self.first_layer(x)
        x = nn.ReLU()(x)
        x = self.dropout_1(x)
        x = self.second_layer(x)
        x = nn.ReLU()(x)
        x = self.dropout_2(x)
        x = self.classifier(x)
        return x

    def __str__(self):
        return f"""Input size: {self.input_size} 
Hidden size 1: {self.hidden_size_1} 
Hidden size 2: {self.hidden_size_2} 
Dropout 1: {self.hidden_dropout_prob_1} 
Dropout 2: {self.hidden_dropout_prob_2} 
Output Size: 2 
"""

    def get_params_string(self):
        return (
            f'dual_output_{self.input_size}_{self.hidden_size_1}_{self.hidden_size_2}_{self.hidden_dropout_prob_1}_{self.hidden_dropout_prob_2}'
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size_1': 4, 'hidden_size_2': 4,
        'hidden_dropout_prob_1': 0.5, 'hidden_dropout_prob_2': 0.5}]
