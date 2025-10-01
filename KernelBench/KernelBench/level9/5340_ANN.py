import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, use_tanh=True):
        super(ANN, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.use_tanh = use_tanh

    def forward(self, encoded_out):
        decoded_out = F.relu(self.l1(encoded_out))
        decoded_out = self.l2(decoded_out)
        if self.use_tanh:
            decoded_out = F.tanh(decoded_out)
        return decoded_out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'hidden_dim': 4}]
