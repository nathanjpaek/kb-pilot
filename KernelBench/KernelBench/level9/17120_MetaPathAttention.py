import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaPathAttention(nn.Module):

    def __init__(self, att_size, latent_dim, metapath_type_num):
        super(MetaPathAttention, self).__init__()
        self.att_size = att_size
        self.latent_dim = latent_dim
        self.metapath_type_num = metapath_type_num
        self.dense_layer_1 = nn.Linear(in_features=latent_dim * 3,
            out_features=att_size)
        self.dense_layer_2 = nn.Linear(in_features=att_size, out_features=1)
        nn.init.xavier_normal_(self.dense_layer_1.weight.data)
        nn.init.xavier_normal_(self.dense_layer_2.weight.data)
        self.lam1 = lambda x, index: x[:, index, :]
        self.lam2 = lambda x: F.softmax(x, dim=1)
        self.lam3 = lambda metapath_latent, atten: torch.sum(
            metapath_latent * torch.unsqueeze(atten, -1), 1)

    def forward(self, user_latent, item_latent, metapath_latent):
        metapath = self.lam1(metapath_latent, 0)
        inputs = torch.cat((user_latent, item_latent, metapath), 1)
        output = self.dense_layer_1(inputs)
        output = F.relu(output)
        output = self.dense_layer_2(output)
        output = F.relu(output)
        for i in range(1, self.metapath_type_num):
            metapath = self.lam1(metapath_latent, i)
            inputs = torch.cat((user_latent, item_latent, metapath), 1)
            tmp_output = self.dense_layer_1(inputs)
            tmp_output = F.relu(tmp_output)
            tmp_output = self.dense_layer_2(tmp_output)
            tmp_output = F.relu(tmp_output)
            output = torch.cat((output, tmp_output), 1)
        atten = self.lam2(output)
        output = self.lam3(metapath_latent, atten)
        return output


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'att_size': 4, 'latent_dim': 4, 'metapath_type_num': 4}]
