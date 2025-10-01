import torch
import torch.nn.functional as F
import torch.nn as nn


class GraphLevelAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features):
        super(GraphLevelAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.my_coefs = None
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.q = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.q.data, gain=1.414)
        self.Tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, total_embeds, P=2):
        h = torch.mm(total_embeds, self.W)
        h_prime = self.Tanh(h + self.b.repeat(h.size()[0], 1))
        semantic_attentions = torch.mm(h_prime, torch.t(self.q)).view(P, -1)
        N = semantic_attentions.size()[1]
        semantic_attentions = semantic_attentions.mean(dim=1, keepdim=True)
        semantic_attentions = F.softmax(semantic_attentions, dim=0)
        self.my_coefs = semantic_attentions
        semantic_attentions = semantic_attentions.view(P, 1, 1)
        semantic_attentions = semantic_attentions.repeat(1, N, self.in_features
            )
        input_embedding = total_embeds.view(P, N, self.in_features)
        h_embedding = torch.mul(input_embedding, semantic_attentions)
        h_embedding = torch.sum(h_embedding, dim=0).squeeze()
        return h_embedding

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
