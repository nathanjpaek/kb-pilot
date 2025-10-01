import torch
import numpy as np
import torch.nn as nn


class MemoryLayer(nn.Module):

    def __init__(self, input_dim, memory_dim, model_dim, mlp_dim, dropout_p):
        super(MemoryLayer, self).__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.model_dim = model_dim
        self.mlp_dim = mlp_dim
        self.W_q = nn.Linear(input_dim, model_dim)
        self.W_k = nn.Linear(memory_dim, model_dim)
        self.W_v = nn.Linear(memory_dim, model_dim)
        self.lin1 = nn.Linear(model_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, input_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()

    def forward(self, x, m):
        Q = self.W_q(x)
        K = self.W_k(m)
        V = self.W_v(m)
        attn = torch.matmul(Q, K.permute(0, 2, 1))
        attn = attn / np.sqrt(self.model_dim)
        attn = self.softmax(attn)
        V_bar = torch.matmul(attn, V)
        out = self.lin1(V_bar)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.lin2(out)
        return out, attn


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'memory_dim': 4, 'model_dim': 4, 'mlp_dim':
        4, 'dropout_p': 0.5}]
