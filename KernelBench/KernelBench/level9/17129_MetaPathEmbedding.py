import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaPathEmbedding(nn.Module):

    def __init__(self, path_num, hop_num, feature_size, latent_dim):
        super(MetaPathEmbedding, self).__init__()
        self.path_num = path_num
        self.hop_num = hop_num
        self.feature_size = feature_size
        self.latent_dim = latent_dim
        self.lam = lambda x, index: x[:, index, :, :]
        if hop_num == 3:
            kernel_size = 3
        elif hop_num == 4:
            kernel_size = 4
        else:
            raise Exception('Only support 3-hop or 4-hop metapaths, hop %d' %
                hop_num)
        self.conv1D = nn.Conv1d(in_channels=self.feature_size, out_channels
            =self.latent_dim, kernel_size=kernel_size, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv1D.weight.data)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input):
        input = input.view((-1, self.path_num, self.hop_num, self.feature_size)
            )
        path_input = self.lam(input, 0)
        path_input = path_input.permute(0, 2, 1)
        output = self.conv1D(path_input).permute(0, 2, 1)
        output = F.relu(output)
        output = self.dropout(output)
        for i in range(1, self.path_num):
            path_input = self.lam(input, i)
            path_input = path_input.permute(0, 2, 1)
            tmp_output = self.conv1D(path_input).permute(0, 2, 1)
            tmp_output = F.relu(tmp_output)
            tmp_output = self.dropout(tmp_output)
            output = torch.cat((output, tmp_output), 2)
        output = output.view((-1, self.path_num, self.latent_dim))
        output = torch.max(output, 1, keepdim=True)[0]
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'path_num': 4, 'hop_num': 4, 'feature_size': 4,
        'latent_dim': 4}]
