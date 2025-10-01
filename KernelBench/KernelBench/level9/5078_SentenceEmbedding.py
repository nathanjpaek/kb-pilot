import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BaseSelfAttention(nn.Module):

    def __init__(self):
        super(BaseSelfAttention, self).__init__()

    def init_linear(self, input_linear):
        """Initialize linear transformation"""
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.
            weight.size(1)))
        nn.init.uniform_(input_linear.weight, -bias, bias)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()

    def initialize_layers(self):
        raise NotImplementedError

    def forward(self, X):
        raise NotImplementedError

    def score(self, a, b):
        raise NotImplementedError


class SentenceEmbedding(BaseSelfAttention):

    def __init__(self, embedding_dim, hidden_dim, num_annotations):
        super(SentenceEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_annotations = num_annotations
        self.initialize_layers()

    def initialize_layers(self):
        self.Ws1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.Ws2 = nn.Linear(self.hidden_dim, self.num_annotations)
        self.init_linear(self.Ws1)
        self.init_linear(self.Ws2)

    def forward(self, word_embeddings):
        """
        Args:
            word_embeddings: 
                (batch_size, doc_maxlen, embedding_dim)
        Output:
            sentence_embedding: 
                (batch_size, num_annotations, embedding_dim)
        """
        hidden = F.tanh(self.Ws1(word_embeddings))
        atten_weights = F.softmax(self.Ws2(hidden), dim=2)
        atten_weights = atten_weights.transpose(1, 2)
        sentence_embedding = atten_weights.bmm(word_embeddings)
        return sentence_embedding


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'embedding_dim': 4, 'hidden_dim': 4, 'num_annotations': 4}]
