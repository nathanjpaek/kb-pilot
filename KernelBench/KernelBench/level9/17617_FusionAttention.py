import torch
import torch.nn.functional as F
import torch.nn as nn


class FusionAttention(nn.Module):

    def __init__(self, dim):
        super(FusionAttention, self).__init__()
        self.attention_matrix = nn.Linear(dim, dim)
        self.project_weight = nn.Linear(dim, 1)

    def forward(self, inputs):
        query_project = self.attention_matrix(inputs)
        query_project = F.leaky_relu(query_project)
        project_value = self.project_weight(query_project)
        attention_weight = torch.softmax(project_value, dim=1)
        attention_vec = inputs * attention_weight
        attention_vec = torch.sum(attention_vec, dim=1)
        return attention_vec, attention_weight


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
