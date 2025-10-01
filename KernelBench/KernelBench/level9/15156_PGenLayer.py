import torch
import torch.nn as nn
import torch.nn.functional as F


class PGenLayer(nn.Module):

    def __init__(self, emb_dim, hidden_size, enc_dim):
        super(PGenLayer, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.enc_dim = enc_dim
        self.lin = nn.Linear(self.emb_dim + self.hidden_size + self.enc_dim, 1)

    def forward(self, emb, hid, enc):
        """
        param:  emb (batch_size, emb_dim)
                hid (batch_size, hid_dim)
                enc (batch_size, enc_dim)
        """
        input = torch.cat((emb, hid, enc), 1)
        return F.sigmoid(self.lin(input))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'emb_dim': 4, 'hidden_size': 4, 'enc_dim': 4}]
