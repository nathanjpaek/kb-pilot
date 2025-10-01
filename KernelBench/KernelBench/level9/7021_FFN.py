import math
import torch
import torch.nn as nn


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    came from : https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/utils/gelu.py
    """

    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 
            0.044715 * torch.pow(x, 3))))


class FFN(nn.Module):

    def __init__(self, embedding_dim, hidden_unit, dropout=0.0, eps=1e-08):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_unit, bias=False)
        self.fc2 = nn.Linear(hidden_unit, embedding_dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.low_ln = nn.LayerNorm(embedding_dim, eps=eps)
        self.mid_ln = nn.LayerNorm(embedding_dim, eps=eps)
        self.hig_ln = nn.LayerNorm(embedding_dim, eps=eps)
        self.act = GELU()

    def forward(self, low_vectors, mid_vectors, hig_vectors):
        low_num, mid_num, hig_num = low_vectors.size()[1], mid_vectors.size()[1
            ], hig_vectors.size()[1]
        low_residual = low_vectors
        mid_residual = mid_vectors
        hig_residual = hig_vectors
        cated_vectors = torch.cat((low_vectors, mid_vectors, hig_vectors),
            dim=1)
        output = self.dropout2(self.fc2(self.dropout1(self.act(self.fc1(
            cated_vectors)))))
        low_vectors, mid_vectors, hig_vectors = torch.split(output, [
            low_num, mid_num, hig_num], dim=1)
        low_vectors = low_residual + low_vectors
        mid_vectors = mid_residual + mid_vectors
        hig_vectors = hig_residual + hig_vectors
        low_vectors = self.low_ln(low_vectors)
        mid_vectors = self.mid_ln(mid_vectors)
        hig_vectors = self.hig_ln(hig_vectors)
        return low_vectors, mid_vectors, hig_vectors


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'embedding_dim': 4, 'hidden_unit': 4}]
