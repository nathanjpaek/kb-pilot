import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionModule(nn.Module):

    def __init__(self):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, A):
        batchsize, num_channels, height, width = A.shape
        N = height * width
        A1 = A.view((batchsize, num_channels, N))
        X = F.softmax(torch.bmm(A1, A1.permute(0, 2, 1)), dim=-1)
        XA1 = torch.bmm(X.permute(0, 2, 1), A1).view((batchsize,
            num_channels, height, width))
        E = self.beta * XA1 + A
        return E

    def initialize_weights(self):
        nn.init.constant_(self.beta.data, 0.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
