import math
import torch
import torch.nn as nn
import torch.nn.init as init


class VocabGraphConvolution(nn.Module):
    """Vocabulary GCN module.

    Params:
        `voc_dim`: The size of vocabulary graph
        `num_adj`: The number of the adjacency matrix of Vocabulary graph
        `hid_dim`: The hidden dimension after XAW
        `out_dim`: The output dimension after Relu(XAW)W
        `dropout_rate`: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.

    Inputs:
        `vocab_adj_list`: The list of the adjacency matrix
        `X_dv`: the feature of mini batch document, can be TF-IDF (batch, vocab), or word embedding (batch, word_embedding_dim, vocab)

    Outputs:
        The graph embedding representation, dimension (batch, `out_dim`) or (batch, word_embedding_dim, `out_dim`)

    """

    def __init__(self, voc_dim, num_adj, hid_dim, out_dim, dropout_rate=0.2):
        super(VocabGraphConvolution, self).__init__()
        self.voc_dim = voc_dim
        self.num_adj = num_adj
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        for i in range(self.num_adj):
            setattr(self, 'W%d_vh' % i, nn.Parameter(torch.randn(voc_dim,
                hid_dim)))
        self.fc_hc = nn.Linear(hid_dim, out_dim)
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if n.startswith('W') or n.startswith('a') or n in ('W', 'a',
                'dense'):
                init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, vocab_adj_list, X_dv, add_linear_mapping_term=False):
        for i in range(self.num_adj):
            H_vh = vocab_adj_list[i].mm(getattr(self, 'W%d_vh' % i))
            H_vh = self.dropout(H_vh)
            H_dh = X_dv.matmul(H_vh)
            if add_linear_mapping_term:
                H_linear = X_dv.matmul(getattr(self, 'W%d_vh' % i))
                H_linear = self.dropout(H_linear)
                H_dh += H_linear
            if i == 0:
                fused_H = H_dh
            else:
                fused_H += H_dh
        out = self.fc_hc(fused_H)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'voc_dim': 4, 'num_adj': 4, 'hid_dim': 4, 'out_dim': 4}]
