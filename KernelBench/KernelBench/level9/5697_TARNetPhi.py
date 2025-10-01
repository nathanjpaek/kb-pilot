import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class TARNetPhi(nn.Module):

    def __init__(self, input_nodes, shared_nodes=200):
        super(TARNetPhi, self).__init__()
        self.shared1 = nn.Linear(in_features=input_nodes, out_features=
            shared_nodes)
        self.shared2 = nn.Linear(in_features=shared_nodes, out_features=
            shared_nodes)
        self.shared3 = nn.Linear(in_features=shared_nodes, out_features=
            shared_nodes)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float()
        else:
            x = x.float()
        x = F.elu(self.shared1(x))
        x = F.elu(self.shared2(x))
        x = F.elu(self.shared3(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_nodes': 4}]
