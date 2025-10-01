import torch
import torch.nn.functional as F
import torch.nn as nn


class SAGEConv(nn.Module):
    """

    Description
    -----------
    SAGE convolutional layer.

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    pool_features : int
        Dimension of pooling features.
    out_features : int
        Dimension of output features.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``None``.
    dropout : float, optional
            Rate of dropout. Default: ``0.0``.
    mu : float, optional
        Hyper-parameter, refer to original paper. Default: ``2.0``.

    """

    def __init__(self, in_features, pool_features, out_features, activation
        =None, mu=2.0, dropout=0.0):
        super(SAGEConv, self).__init__()
        self.pool_layer = nn.Linear(in_features, pool_features)
        self.linear1 = nn.Linear(pool_features, out_features)
        self.linear2 = nn.Linear(pool_features, out_features)
        self.activation = activation
        self.mu = mu
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        if self.activation == F.leaky_relu:
            gain = nn.init.calculate_gain('leaky_relu')
        else:
            gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear1.weight, gain=gain)
        nn.init.xavier_normal_(self.linear2.weight, gain=gain)
        nn.init.xavier_normal_(self.pool_layer.weight, gain=gain)

    def forward(self, x, adj):
        """

        Parameters
        ----------
        x : torch.Tensor
            Tensor of input features.
        adj : torch.SparseTensor
            Sparse tensor of adjacency matrix.


        Returns
        -------
        x : torch.Tensor
            Output of layer.

        """
        x = F.relu(self.pool_layer(x))
        x_ = x ** self.mu
        x_ = torch.spmm(adj, x_) ** (1 / self.mu)
        x = self.linear1(x)
        x_ = self.linear2(x_)
        x = x + x_
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'pool_features': 4, 'out_features': 4}]
