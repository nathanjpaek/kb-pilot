import torch
import torch.multiprocessing
from torch import nn
import torch.utils.data


class SingleGate(nn.Module):

    def __init__(self, vector_dim, topic_dim):
        super().__init__()
        assert vector_dim == topic_dim
        self.fusion_linear = nn.Linear(vector_dim + topic_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, source_vector, other_vector):
        context_input = torch.cat((source_vector, other_vector), dim=1)
        context_gate = self.sigmoid(self.fusion_linear(context_input))
        context_fusion = context_gate * source_vector + (1.0 - context_gate
            ) * other_vector
        return context_fusion


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'vector_dim': 4, 'topic_dim': 4}]
