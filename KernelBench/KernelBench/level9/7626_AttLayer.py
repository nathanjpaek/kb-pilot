import torch
import torch.nn as nn
import torch.nn.functional as fn


class AttLayer(nn.Module):
    """Calculate the attention signal(weight) according the input tensor.

    Args:
        infeatures (torch.FloatTensor): A 3D input tensor with shape of[batch_size, M, embed_dim].

    Returns:
        torch.FloatTensor: Attention weight of input. shape of [batch_size, M].
    """

    def __init__(self, in_dim, att_dim):
        super(AttLayer, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.w = torch.nn.Linear(in_features=in_dim, out_features=att_dim,
            bias=False)
        self.h = nn.Parameter(torch.randn(att_dim), requires_grad=True)

    def forward(self, infeatures):
        att_singal = self.w(infeatures)
        att_singal = fn.relu(att_singal)
        att_singal = torch.mul(att_singal, self.h)
        att_singal = torch.sum(att_singal, dim=2)
        att_singal = fn.softmax(att_singal, dim=1)
        return att_singal


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'att_dim': 4}]
