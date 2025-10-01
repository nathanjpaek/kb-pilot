import torch
from torch import nn


class Gate(nn.Module):

    def __init__(self, input_size, dropout=0.2):
        """ To determine the importance of passage parts and
            attend to the ones relevant to the question, this Gate was added
            to the input of RNNCell in both Gated Attention-based Recurrent
            Network and Self-Matching Attention.

            Args:
                input_size(int): size of input vectors
                dropout (float, optional): dropout probability

            Input:
                - **input** of shape `(batch, input_size)`: a float tensor containing concatenated
                  passage representation and attention vector both calculated for each word in the passage

            Output:
                - **output** of shape `(batch, input_size)`: a float tensor containing gated input
        """
        super(Gate, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.W = nn.Linear(input_size, input_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        result = self.W(input)
        self.dropout(result)
        result = self.sigmoid(result)
        return result * input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
