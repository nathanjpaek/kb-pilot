import torch
import torch.nn as nn
import torch.optim


class Merge(nn.Module):

    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)
        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding,
            sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding,
            sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'embedding_size': 4}]
