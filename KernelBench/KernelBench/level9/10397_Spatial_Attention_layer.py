import torch
import torch.nn.functional as F
import torch.nn as nn


class Spatial_Attention_layer(nn.Module):
    """
    compute spatial attention scores
    """

    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps)
            )
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices,
            num_of_vertices))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices,
            num_of_vertices))

    def forward(self, x):
        """
        Making a forward pass of the spatial attention layer.
        B is the batch size. N_nodes is the number of nodes in the graph. F_in is the dimension of input features. 
        T_in is the length of input sequence in time. 
        Arg types:
            * x (PyTorch Float Tensor)* - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).

        Return types:
            * output (PyTorch Float Tensor)* - Spatial attention score matrices, with shape (B, N_nodes, N_nodes).
        """
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)
        product = torch.matmul(lhs, rhs)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        S_normalized = F.softmax(S, dim=1)
        return S_normalized


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'DEVICE': 4, 'in_channels': 4, 'num_of_vertices': 4,
        'num_of_timesteps': 4}]
