import torch
from typing import Optional
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Union


class Balance_Theory(nn.Module):
    """The signed graph clustering model with balance theory, restricted to 2 hops for fair comparison with SSSNET.

    Args:
        nfeat (int): Number of features.
        hidden (int): Hidden dimensions of the initial MLP.
        nclass (int): Number of clusters.
        dropout (float): Dropout probability.
        hop (int): Number of hops to consider. (need to be 2)
        directed (bool, optional): Whether the input network is directed or not. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, nfeat: 'int', hidden: 'int', nclass: 'int', dropout:
        'float', hop: 'int', directed: 'bool'=False, bias: 'bool'=True):
        super(Balance_Theory, self).__init__()
        nh1 = hidden
        nh2 = hidden
        self._num_clusters = int(nclass)
        assert hop == 2, 'please only use 2 hops'
        self._hop_p = 4
        self._hop_n = 3
        if bias:
            self._bias = Parameter(torch.FloatTensor(self._num_clusters))
        else:
            self.register_parameter('_bias', None)
        self._relu = nn.ReLU()
        self._dropout = nn.Dropout(p=dropout)
        self._undirected = not directed
        if self._undirected:
            self._w_p0 = Parameter(torch.FloatTensor(nfeat, nh1))
            self._w_p1 = Parameter(torch.FloatTensor(nh1, nh2))
            self._w_n0 = Parameter(torch.FloatTensor(nfeat, nh1))
            self._w_n1 = Parameter(torch.FloatTensor(nh1, nh2))
            self._w_p = Parameter(torch.FloatTensor(self._hop_p, 1))
            self._w_n = Parameter(torch.FloatTensor(self._hop_n, 1))
            self._W_prob = Parameter(torch.FloatTensor(2 * nh2, self.
                _num_clusters))
            self._reset_parameters_undirected()
        else:
            self._w_sp0 = Parameter(torch.FloatTensor(nfeat, nh1))
            self._w_sp1 = Parameter(torch.FloatTensor(nh1, nh2))
            self._w_sn0 = Parameter(torch.FloatTensor(nfeat, nh1))
            self._w_sn1 = Parameter(torch.FloatTensor(nh1, nh2))
            self._w_tp0 = Parameter(torch.FloatTensor(nfeat, nh1))
            self._w_tp1 = Parameter(torch.FloatTensor(nh1, nh2))
            self._w_tn0 = Parameter(torch.FloatTensor(nfeat, nh1))
            self._w_tn1 = Parameter(torch.FloatTensor(nh1, nh2))
            self._w_sp = Parameter(torch.FloatTensor(self._hop_p, 1))
            self._w_sn = Parameter(torch.FloatTensor(self._hop_n, 1))
            self._w_tp = Parameter(torch.FloatTensor(self._hop_p, 1))
            self._w_tn = Parameter(torch.FloatTensor(self._hop_n, 1))
            self._W_prob = Parameter(torch.FloatTensor(4 * nh2, self.
                _num_clusters))
            self._reset_parameters_directed()

    def _reset_parameters_undirected(self):
        self._w_p.data.fill_(1.0)
        self._w_n.data.fill_(1.0)
        nn.init.xavier_uniform_(self._w_p0, gain=1.414)
        nn.init.xavier_uniform_(self._w_p1, gain=1.414)
        nn.init.xavier_uniform_(self._w_n0, gain=1.414)
        nn.init.xavier_uniform_(self._w_n1, gain=1.414)
        if self._bias is not None:
            self._bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self._W_prob, gain=1.414)

    def _reset_parameters_directed(self):
        self._w_sp.data.fill_(1.0)
        self._w_sn.data.fill_(1.0)
        self._w_tp.data.fill_(1.0)
        self._w_tn.data.fill_(1.0)
        nn.init.xavier_uniform_(self._w_sp0, gain=1.414)
        nn.init.xavier_uniform_(self._w_sp1, gain=1.414)
        nn.init.xavier_uniform_(self._w_sn0, gain=1.414)
        nn.init.xavier_uniform_(self._w_sn1, gain=1.414)
        nn.init.xavier_uniform_(self._w_tp0, gain=1.414)
        nn.init.xavier_uniform_(self._w_tp1, gain=1.414)
        nn.init.xavier_uniform_(self._w_tn0, gain=1.414)
        nn.init.xavier_uniform_(self._w_tn1, gain=1.414)
        if self._bias is not None:
            self._bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self._W_prob, gain=1.414)

    def forward(self, A_p:
        'Union[torch.FloatTensor, torch.sparse_coo_tensor]', A_n:
        'Union[torch.FloatTensor, torch.sparse_coo_tensor]', features:
        'torch.FloatTensor', A_pt:
        'Optional[Union[torch.FloatTensor, torch.sparse_coo_tensor]]'=None,
        A_nt: 'Optional[Union[torch.FloatTensor, torch.sparse_coo_tensor]]'
        =None) ->Tuple[torch.FloatTensor, torch.FloatTensor, torch.
        LongTensor, torch.FloatTensor]:
        """
        Making a forward pass of the signed graph clustering model with balance theory.
        
        Arg types:
            * **A_p** (PyTorch FloatTensor or PyTorch sparse_coo_tensor) - Row-normalized positive part of the adjacency matrix.
            * **A_n** (PyTorch FloatTensor or PyTorch sparse_coo_tensor) - Row-normalized negative part of the adjacency matrix.
            * **features** (PyTorch FloatTensor) - Input node features, with shape (num_nodes, num_features).
            * **A_pt** (PyTorch FloatTensor or PyTorch sparse_coo_tensor, optional) - Transpose of column-normalized 
                positive part of the adjacency matrix. Default: None.
            * **A_nt** (PyTorch FloatTensor or PyTorch sparse_coo_tensor, optional) - Transpose of column-normalized 
                negative part of the adjacency matrix. Default: None.

        Return types:
            * **z** (PyTorch FloatTensor) - Embedding matrix, with shape (num_nodes, 2*hidden) for undirected graphs 
                and (num_nodes, 4*hidden) for directed graphs.
            * **output** (PyTorch FloatTensor) - Log of prob, with shape (num_nodes, num_clusters).
            * **predictions_cluster** (PyTorch LongTensor) - Predicted labels.
            * **prob** (PyTorch FloatTensor) - Probability assignment matrix of different clusters, with shape (num_nodes, num_clusters).
        """
        if self._undirected:
            x_p = torch.mm(features, self._w_p0)
            x_p = self._relu(x_p)
            x_p = self._dropout(x_p)
            x_p = torch.mm(x_p, self._w_p1)
            x_n = torch.mm(features, self._w_n0)
            x_n = self._relu(x_n)
            x_n = self._dropout(x_n)
            x_n = torch.mm(x_n, self._w_n1)
            feat_p = self._w_p[0] * x_p
            feat_n = torch.zeros_like(feat_p)
            curr_p = x_p.clone()
            curr_n_aux = x_n.clone()
            j = 0
            for h in range(0, self._hop_p - 1):
                if h > 0:
                    curr_p = torch.matmul(A_p, curr_p)
                    curr_n_aux = torch.matmul(A_p, curr_n_aux)
                    feat_p += self._w_p[h] * curr_p
                if h != self._hop_p - 2:
                    curr_n = torch.matmul(A_n, curr_n_aux)
                    feat_n += self._w_n[j] * curr_n
                    j += 1
                    for _ in range(self._hop_p - 3 - h):
                        curr_n = torch.matmul(A_p, curr_n)
                        feat_n += self._w_n[j] * curr_n
                        j += 1
            feat_p += self._w_p[3] * torch.matmul(A_n, torch.matmul(A_n, x_p))
            feat = torch.cat([feat_p, feat_n], dim=1)
        else:
            x_sp = torch.mm(features, self._w_sp0)
            x_sp = self._relu(x_sp)
            x_sp = self._dropout(x_sp)
            x_sp = torch.mm(x_sp, self._w_sp1)
            x_sn = torch.mm(features, self._w_sn0)
            x_sn = self._relu(x_sn)
            x_sn = self._dropout(x_sn)
            x_sn = torch.mm(x_sn, self._w_sn1)
            x_tp = torch.mm(features, self._w_tp0)
            x_tp = self._relu(x_tp)
            x_tp = self._dropout(x_tp)
            x_tp = torch.mm(x_tp, self._w_tp1)
            x_tn = torch.mm(features, self._w_tn0)
            x_tn = self._relu(x_tn)
            x_tn = self._dropout(x_tn)
            x_tn = torch.mm(x_tn, self._w_tn1)
            A_sp = A_p
            A_sn = A_n
            A_tp = A_pt
            A_tn = A_nt
            feat_sp = self._w_sp[0] * x_sp
            feat_sn = torch.zeros_like(feat_sp)
            feat_tp = self._w_tp[0] * x_tp
            feat_tn = torch.zeros_like(feat_tp)
            curr_sp = x_sp.clone()
            curr_sn_aux = x_sn.clone()
            curr_tp = x_tp.clone()
            curr_tn_aux = x_tn.clone()
            j = 0
            for h in range(0, self._hop_p - 1):
                if h > 0:
                    curr_sp = torch.matmul(A_sp, curr_sp)
                    curr_sn_aux = torch.matmul(A_sp, curr_sn_aux)
                    curr_tp = torch.matmul(A_tp, curr_tp)
                    curr_tn_aux = torch.matmul(A_tp, curr_tn_aux)
                    feat_sp += self._w_sp[h] * curr_sp
                    feat_tp += self._w_tp[h] * curr_tp
                if h != self._hop_p - 2:
                    curr_sn = torch.matmul(A_sn, curr_sn_aux)
                    curr_tn = torch.matmul(A_tn, curr_tn_aux)
                    feat_sn += self._w_sn[j] * curr_sn
                    feat_tn += self._w_tn[j] * curr_tn
                    j += 1
                    for _ in range(self._hop_p - 3 - h):
                        curr_sn = torch.matmul(A_sp, curr_sn)
                        curr_tn = torch.matmul(A_tp, curr_tn)
                        feat_sn += self._w_sn[j] * curr_sn
                        feat_tn += self._w_tn[j] * curr_tn
                        j += 1
            feat_sp += self._w_sp[3] * torch.matmul(A_sn, torch.matmul(A_sn,
                x_sp))
            feat_tp += self._w_tp[3] * torch.matmul(A_tn, torch.matmul(A_tn,
                x_tp))
            feat = torch.cat([feat_sp, feat_sn, feat_tp, feat_tn], dim=1)
        z = feat
        output = torch.mm(z, self._W_prob)
        if self._bias is not None:
            output = output + self._bias
        predictions_cluster = torch.argmax(output, dim=1)
        prob = F.softmax(output, dim=1)
        output = F.log_softmax(output, dim=1)
        return F.normalize(z), output, predictions_cluster, prob


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'nfeat': 4, 'hidden': 4, 'nclass': 4, 'dropout': 0.5,
        'hop': 2}]
