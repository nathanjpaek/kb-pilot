import torch
import torch.nn as nn
import torch.nn.functional as F


class UIAttention(nn.Module):

    def __init__(self, latent_dim, att_size):
        super(UIAttention, self).__init__()
        self.dense = nn.Linear(in_features=latent_dim * 2, out_features=
            att_size)
        nn.init.xavier_normal_(self.dense.weight.data)
        self.lam = lambda x: F.softmax(x, dim=1)

    def forward(self, input, path_output):
        inputs = torch.cat((input, path_output), 1)
        output = self.dense(inputs)
        output = torch.relu(output)
        atten = self.lam(output)
        output = input * atten
        return output


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'latent_dim': 4, 'att_size': 4}]
