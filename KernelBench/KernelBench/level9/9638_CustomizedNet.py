import torch
import torch.nn as nn
import torch.utils.data.distributed


class CustomizedNet(nn.Module):

    def __init__(self, dropout, input_size, input_feature_num, hidden_dim,
        output_size):
        """
        Simply use linear layers for multi-variate single-step forecasting.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size * input_feature_num, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = torch.unsqueeze(x, 1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dropout': 0.5, 'input_size': 4, 'input_feature_num': 4,
        'hidden_dim': 4, 'output_size': 4}]
