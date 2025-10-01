import torch
import torch.nn as nn


class DummyEmbedder(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.day_embedding = nn.Linear(1, embedding_dim)
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.year_embedding = nn.Linear(1, embedding_dim)
        self.dummy_fusion = nn.Linear(embedding_dim * 4, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, temporal_features):
        d, w, m, y = temporal_features[:, 0].unsqueeze(1), temporal_features[
            :, 1].unsqueeze(1), temporal_features[:, 2].unsqueeze(1
            ), temporal_features[:, 3].unsqueeze(1)
        d_emb, w_emb, m_emb, y_emb = self.day_embedding(d
            ), self.week_embedding(w), self.month_embedding(m
            ), self.year_embedding(y)
        temporal_embeddings = self.dummy_fusion(torch.cat([d_emb, w_emb,
            m_emb, y_emb], dim=1))
        temporal_embeddings = self.dropout(temporal_embeddings)
        return temporal_embeddings


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'embedding_dim': 4}]
