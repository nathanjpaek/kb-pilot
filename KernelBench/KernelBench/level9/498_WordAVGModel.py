import torch
import torch.nn as nn
import torch.nn.functional as F


class WordAVGModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, int((embedding_dim + output_dim
            ) / 2))
        self.fc2 = nn.Linear(int((embedding_dim + output_dim) / 2), output_dim)
        self.dropout = nn.Dropout(dropout)
    """ seq: [batch_size, seq_size]"""

    def forward(self, text):
        embedded = self.embedding(text.long())
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze()
        output = self.dropout(self.fc1(pooled))
        return self.fc2(output)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'vocab_size': 4, 'embedding_dim': 4, 'output_dim': 4}]
