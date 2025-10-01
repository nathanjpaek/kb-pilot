import torch
import torch.utils.data
from torch import nn


class FFNN1(nn.Module):

    def __init__(self, input_size, hidden_size, hidden_dropout_prob):
        super(FFNN1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.first_layer = nn.Linear(input_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_layer(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def __str__(self):
        return f"""Input size: {self.input_size} 
Hidden size: {self.hidden_size} 
Dropout: {self.hidden_dropout_prob} 
Output Size: 1 
"""

    def get_params_string(self):
        return (
            f'{self.input_size}_{self.hidden_size}_{self.hidden_dropout_prob_1}'
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'hidden_dropout_prob': 0.5}
        ]
