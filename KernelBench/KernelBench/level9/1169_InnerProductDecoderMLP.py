import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.modules.loss
import torch.utils.data


class InnerProductDecoderMLP(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout, act=
        torch.sigmoid):
        super(InnerProductDecoderMLP, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout = dropout
        self.act = act
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, z):
        z = F.relu(self.fc(z))
        z = torch.sigmoid(self.fc2(z))
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.bmm(z, torch.transpose(z, 1, 2)))
        return adj


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim1': 4, 'hidden_dim2': 4,
        'dropout': 0.5}]
