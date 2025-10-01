import torch
import torch.nn as nn
import torch.utils.data


class Readout(nn.Module):
    """
    This module learns a single graph level representation for a molecule given GraphSAGE generated embeddings
    """

    def __init__(self, attr_dim, embedding_dim, hidden_dim, output_dim,
        num_cats):
        super(Readout, self).__init__()
        self.attr_dim = attr_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_cats = num_cats
        self.layer1 = nn.Linear(attr_dim + embedding_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.output = nn.Linear(output_dim, num_cats)
        self.act = nn.ReLU()

    def forward(self, node_features, node_embeddings):
        combined_rep = torch.cat((node_features, node_embeddings), dim=1)
        hidden_rep = self.act(self.layer1(combined_rep))
        graph_rep = self.act(self.layer2(hidden_rep))
        logits = torch.mean(self.output(graph_rep), dim=0)
        return logits


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'attr_dim': 4, 'embedding_dim': 4, 'hidden_dim': 4,
        'output_dim': 4, 'num_cats': 4}]
