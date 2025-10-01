import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.distributions


class InformedSender(nn.Module):

    def __init__(self, game_size, feat_size, embedding_size, hidden_size,
        vocab_size=100, temp=1.0):
        super(InformedSender, self).__init__()
        self.game_size = game_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.temp = temp
        self.lin1 = nn.Linear(feat_size, embedding_size, bias=False)
        self.conv2 = nn.Conv2d(1, hidden_size, kernel_size=(game_size, 1),
            stride=(game_size, 1), bias=False)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=(hidden_size, 1), stride=(
            hidden_size, 1), bias=False)
        self.lin4 = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, x, return_embeddings=False):
        emb = self.return_embeddings(x)
        h = self.conv2(emb)
        h = torch.sigmoid(h)
        h = h.transpose(1, 2)
        h = self.conv3(h)
        h = torch.sigmoid(h)
        h = h.squeeze(dim=1)
        h = h.squeeze(dim=1)
        h = self.lin4(h)
        h = h.mul(1.0 / self.temp)
        logits = F.log_softmax(h, dim=1)
        return logits

    def return_embeddings(self, x):
        embs = []
        for i in range(self.game_size):
            h = x[i]
            if len(h.size()) == 3:
                h = h.squeeze(dim=-1)
            h_i = self.lin1(h)
            h_i = h_i.unsqueeze(dim=1)
            h_i = h_i.unsqueeze(dim=1)
            embs.append(h_i)
        h = torch.cat(embs, dim=2)
        return h


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'game_size': 4, 'feat_size': 4, 'embedding_size': 4,
        'hidden_size': 4}]
