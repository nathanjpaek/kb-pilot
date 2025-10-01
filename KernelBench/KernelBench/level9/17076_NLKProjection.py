import torch
from torch import nn
import torch.nn.functional as F


class TwoLayerNet(nn.Module):

    def __init__(self, dim, hidden_dim, output_dim):
        super(TwoLayerNet, self).__init__()
        self.layer1 = nn.Linear(dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, emb):
        return self.layer2(F.relu(self.layer1(emb)))


class NLKProjection(nn.Module):

    def __init__(self, dim, hidden_dim, group_num):
        super(NLKProjection, self).__init__()
        self.dim, self.hidden_dim, self.concat_dim = (dim, hidden_dim, 2 *
            dim + group_num)
        self.MLP1 = TwoLayerNet(dim, hidden_dim, dim)
        self.MLP2 = TwoLayerNet(dim, hidden_dim, dim)
        self.MLP3 = TwoLayerNet(self.concat_dim, hidden_dim, dim)
        self.MLP4 = TwoLayerNet(self.concat_dim, hidden_dim, dim)

    def forward(self, origin_center, origin_offset, x_new):
        z1 = self.MLP1(origin_center)
        z2 = self.MLP2(origin_offset)
        final_input = torch.cat([z1, z2, x_new], dim=-1)
        new_offset = self.MLP3(final_input)
        new_center = self.MLP4(final_input)
        return torch.cat([new_center, new_offset, x_new], dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'hidden_dim': 4, 'group_num': 4}]
