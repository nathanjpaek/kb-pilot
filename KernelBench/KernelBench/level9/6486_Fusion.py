import torch
import torch.nn as nn
import torch.utils.checkpoint


class Fusion(nn.Module):
    """
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    """

    def __init__(self, in_size, hidden_size, n_class, dropout, modal_name=
        'text'):
        """
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        """
        super(Fusion, self).__init__()
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch_size, in_size)
        """
        dropped = self.drop(x)
        y_1 = torch.tanh(self.linear_1(dropped))
        self.linear_2(y_1)
        y_2 = torch.tanh(self.linear_2(y_1))
        y_3 = self.linear_3(y_2)
        return y_2, y_3


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'hidden_size': 4, 'n_class': 4, 'dropout': 0.5}]
