import torch
import torch.utils.data
import torch.nn as nn


class ItemInferenceNetwork(nn.Module):

    def __init__(self, num_item, item_feat_dim):
        super().__init__()
        self.mu_lookup = nn.Embedding(num_item, item_feat_dim)
        self.logvar_lookup = nn.Embedding(num_item, item_feat_dim)

    def forward(self, item_index):
        item_index = item_index.squeeze(1)
        mu = self.mu_lookup(item_index.long())
        logvar = self.logvar_lookup(item_index.long())
        return mu, logvar


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_item': 4, 'item_feat_dim': 4}]
