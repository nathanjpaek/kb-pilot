import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, X):
        """
        :param X: tensor dimension batch x len_q x d_model
        :return out: tensor dimension batch x len_q x d_model
        """
        out = self.linear2(nn.ReLU()(self.linear1(X)))
        return self.layer_norm(out + X)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_ff': 4}]
