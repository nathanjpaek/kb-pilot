import torch
import torch.nn.functional as F
import torch.nn as nn


class TAGConv(nn.Module):
    """

    Description
    -----------
    TAGCN convolutional layer.

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    k : int, optional
        Hyper-parameter, refer to original paper. Default: ``2``.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``None``.

    batch_norm : bool, optional
        Whether to apply batch normalization. Default: ``False``.
    dropout : float, optional
        Rate of dropout. Default: ``0.0``.

    """

    def __init__(self, in_features, out_features, k=2, activation=None,
        batch_norm=False, dropout=0.0):
        super(TAGConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features * (k + 1), out_features)
        self.batch_norm = batch_norm
        if batch_norm:
            self.norm_func = nn.BatchNorm1d(out_features, affine=False)
        self.activation = activation
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.k = k
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        if self.activation == F.leaky_relu:
            gain = nn.init.calculate_gain('leaky_relu')
        else:
            gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

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
        fstack = [x]
        for i in range(self.k):
            y = torch.spmm(adj, fstack[-1])
            fstack.append(y)
        x = torch.cat(fstack, dim=-1)
        x = self.linear(x)
        if self.batch_norm:
            x = self.norm_func(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
