import torch
from torch.nn import functional as F
import torch.multiprocessing
from torch import nn
import torch.utils.data


class ContextGate(nn.Module):

    def __init__(self, vector_dim, topic_dim):
        super().__init__()
        assert vector_dim == topic_dim
        self.fusion_linear = nn.Linear(vector_dim + topic_dim, vector_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, source_vector, other_vector):
        context_input = torch.cat((source_vector, other_vector), dim=1)
        context_gate = self.sigmoid(self.fusion_linear(context_input))
        context_fusion = context_gate * source_vector + (1.0 - context_gate
            ) * other_vector
        return self.tanh(context_fusion)


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


class Lda2Vec(nn.Module):

    def __init__(self, word_vec_dim, topic_emb_dim, topic_threshold=0.1,
        mode='gate_all'):
        super(Lda2Vec, self).__init__()
        self.mode = mode
        if mode == 'gate_one':
            self.fusion_layer = SingleGate(word_vec_dim, topic_emb_dim)
        elif mode == 'gate_all':
            assert word_vec_dim == topic_emb_dim
            self.fusion_layer = ContextGate(word_vec_dim, topic_emb_dim)
        self.topic_threshold = topic_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, word_embedding, topic_embedding, topic_dist):
        topic_dist = F.softmax(topic_dist, dim=1)
        topic_dist = topic_dist - self.topic_threshold
        topic_hidden = torch.matmul(topic_dist, topic_embedding)
        merge_embedding = self.fusion_layer(word_embedding, topic_hidden)
        return merge_embedding


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'word_vec_dim': 4, 'topic_emb_dim': 4}]
