import torch
import torch.nn as nn
import torch.nn.functional as F


class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """

    def __init__(self, input_size, out_size):
        super(SageLayer, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.linear = nn.Linear(self.input_size * 2, self.out_size)

    def forward(self, self_feats, aggregate_feats, neighs=None):
        """
        Generates embeddings for a batch of nodes.

        nodes    -- list of nodes
        """
        combined = torch.cat([self_feats, aggregate_feats], dim=1)
        combined = F.relu(self.linear(combined))
        return combined


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'out_size': 4}]
