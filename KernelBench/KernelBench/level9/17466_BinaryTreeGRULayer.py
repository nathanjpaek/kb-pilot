import torch
import torch.nn as nn


class BinaryTreeGRULayer(nn.Module):

    def __init__(self, hidden_dim):
        super(BinaryTreeGRULayer, self).__init__()
        self.fc1 = nn.Linear(in_features=2 * hidden_dim, out_features=3 *
            hidden_dim)
        self.fc2 = nn.Linear(in_features=2 * hidden_dim, out_features=
            hidden_dim)

    def forward(self, hl, hr):
        """
        Args:
            hl: (batch_size, max_length, hidden_dim).
            hr: (batch_size, max_length, hidden_dim).
        Returns:
            h: (batch_size, max_length, hidden_dim).
        """
        hlr_cat1 = torch.cat([hl, hr], dim=-1)
        treegru_vector = self.fc1(hlr_cat1)
        i, f, r = treegru_vector.chunk(chunks=3, dim=-1)
        hlr_cat2 = torch.cat([hl * r.sigmoid(), hr * r.sigmoid()], dim=-1)
        h_hat = self.fc2(hlr_cat2)
        h = (hl + hr) * f.sigmoid() + h_hat.tanh() * i.sigmoid()
        return h


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4}]
