import torch
import torch.nn.functional as F
import torch.nn as nn


class Temporal_Attention_layer(nn.Module):

    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps,
            num_of_timesteps))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps,
            num_of_timesteps))

    def forward(self, x):
        """
        Making a forward pass of the temporal attention layer.
        B is the batch size. N_nodes is the number of nodes in the graph. F_in is the dimension of input features. 
        T_in is the length of input sequence in time. 
        Arg types:
            * x (PyTorch Float Tensor)* - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).

        Return types:
            * output (PyTorch Float Tensor)* - Temporal attention score matrices, with shape (B, T_in, T_in).
        """
        _, _num_of_vertices, _num_of_features, _num_of_timesteps = x.shape
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1),
            self.U2)
        rhs = torch.matmul(self.U3, x)
        product = torch.matmul(lhs, rhs)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        E_normalized = F.softmax(E, dim=1)
        return E_normalized


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'DEVICE': 4, 'in_channels': 4, 'num_of_vertices': 4,
        'num_of_timesteps': 4}]
