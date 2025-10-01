import torch
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEncoderDecoderAttentionLayer(nn.Module):
    """
    Graph-to-Graph message passing, adapted from https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_src_features, in_tgt_features, out_features,
        dropout, alpha, concat=True):
        super(GraphEncoderDecoderAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_src_features = in_src_features
        self.in_tgt_features = in_tgt_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.Ws = nn.Parameter(torch.empty(size=(in_src_features,
            out_features)))
        self.Wt = nn.Parameter(torch.empty(size=(in_tgt_features,
            out_features)))
        nn.init.xavier_uniform_(self.Ws.data, gain=1.414)
        nn.init.xavier_uniform_(self.Wt.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, ctx, adj):
        Ws_ctx = torch.bmm(ctx, self.Ws.repeat(ctx.size(0), 1, 1))
        Wt_h = torch.bmm(h, self.Wt.repeat(h.size(0), 1, 1))
        a_input = self._prepare_attentional_mechanism_input(Ws_ctx, Wt_h)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9000000000000000.0 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Ws_ctx)
        h_prime = F.leaky_relu(h_prime)
        return h_prime

    def _prepare_attentional_mechanism_input(self, Ws_ctx, Wt_h):
        Ns = Ws_ctx.size()[1]
        Nt = Wt_h.size()[1]
        Ws_ctx_repeated_in_chunks = Ws_ctx.repeat_interleave(Nt, dim=1)
        Wt_h_repeated_alternating = Wt_h.repeat([1, Ns, 1])
        all_combinations_matrix = torch.cat([Ws_ctx_repeated_in_chunks,
            Wt_h_repeated_alternating], dim=2)
        return all_combinations_matrix.view(Ws_ctx.size(0), Nt, Ns, 2 *
            self.out_features)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'in_src_features': 4, 'in_tgt_features': 4, 'out_features':
        4, 'dropout': 0.5, 'alpha': 4}]
