import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):

    def __init__(self, dim_nd, dim_ft, dim_hd, dim_ot, drop_rate=0.5):
        super(GCN, self).__init__()
        self.lin1 = nn.Linear(dim_ft, dim_hd)
        self.lin2 = nn.Linear(dim_hd, dim_ot)
        self.act1 = F.relu
        self.act2 = nn.Softmax
        self.drop1 = nn.Dropout(p=drop_rate)
        self.drop2 = nn.Dropout(p=drop_rate)

    def forward(self, A, X):
        temp = self.drop1(X)
        temp = torch.sparse.mm(A, temp)
        temp = self.lin1(temp)
        temp = self.act1(temp)
        temp = self.drop2(temp)
        temp = torch.sparse.mm(A, temp)
        temp = self.lin2(temp)
        output = temp
        return output


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dim_nd': 4, 'dim_ft': 4, 'dim_hd': 4, 'dim_ot': 4}]
