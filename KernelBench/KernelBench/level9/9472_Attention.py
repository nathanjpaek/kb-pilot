import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Defining the attention layer to be used with Bi-LSTM"""

    def __init__(self, hidden_dim):
        """Constructor for the Attention class.

			Args:
				hidden_dim (int): The double of the hidden vector size of the LSTM unit. This is because we are using Bi-LSTM so double hidden layer size.
		"""
        super(Attention, self).__init__()
        self.weights = nn.Linear(hidden_dim, hidden_dim)
        self.context_u = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs):
        """Computes the forward pass for the attention layer.
			
			Args:
				lstm_outputs (torch.Tensor): The concatenated forward and backward hidden state output for each of the word in the sentence

			Returns:
				weighted_sum (torch.Tensor): The attention weighted sum for the hidden states.
		"""
        tanh_h = torch.tanh(self.weights(lstm_outputs))
        context_multiplied = self.context_u(tanh_h)
        scores = F.softmax(context_multiplied, dim=1)
        weighted_sum = (scores * lstm_outputs).sum(1)
        return weighted_sum


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4}]
