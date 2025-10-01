import torch
from typing import Optional
from typing import Tuple
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Union


class SIMPA(nn.Module):
    """The signed mixed-path aggregation model.

    Args:
        hop (int): Number of hops to consider.
        directed (bool, optional): Whether the input network is directed or not. (default: :obj:`False`)
    """

    def __init__(self, hop: 'int', directed: 'bool'=False):
        super(SIMPA, self).__init__()
        self._hop_p = hop + 1
        self._hop_n = int((1 + hop) * hop / 2)
        self._undirected = not directed
        if self._undirected:
            self._w_p = Parameter(torch.FloatTensor(self._hop_p, 1))
            self._w_n = Parameter(torch.FloatTensor(self._hop_n, 1))
            self._reset_parameters_undirected()
        else:
            self._w_sp = Parameter(torch.FloatTensor(self._hop_p, 1))
            self._w_sn = Parameter(torch.FloatTensor(self._hop_n, 1))
            self._w_tp = Parameter(torch.FloatTensor(self._hop_p, 1))
            self._w_tn = Parameter(torch.FloatTensor(self._hop_n, 1))
            self._reset_parameters_directed()

    def _reset_parameters_undirected(self):
        self._w_p.data.fill_(1.0)
        self._w_n.data.fill_(1.0)

    def _reset_parameters_directed(self):
        self._w_sp.data.fill_(1.0)
        self._w_sn.data.fill_(1.0)
        self._w_tp.data.fill_(1.0)
        self._w_tn.data.fill_(1.0)

    def forward(self, A_p:
        'Union[torch.FloatTensor, torch.sparse_coo_tensor]', A_n:
        'Union[torch.FloatTensor, torch.sparse_coo_tensor]', x_p:
        'torch.FloatTensor', x_n: 'torch.FloatTensor', x_pt:
        'Optional[torch.FloatTensor]'=None, x_nt:
        'Optional[torch.FloatTensor]'=None, A_pt:
        'Optional[Union[torch.FloatTensor, torch.sparse_coo_tensor]]'=None,
        A_nt: 'Optional[Union[torch.FloatTensor, torch.sparse_coo_tensor]]'
        =None) ->Tuple[torch.FloatTensor, torch.FloatTensor, torch.
        LongTensor, torch.FloatTensor]:
        """
        Making a forward pass of SIMPA.
        
        Arg types:
            * **A_p** (PyTorch FloatTensor or PyTorch sparse_coo_tensor) - Row-normalized positive part of the adjacency matrix.
            * **A_n** (PyTorch FloatTensor or PyTorch sparse_coo_tensor) - Row-normalized negative part of the adjacency matrix.
            * **x_p** (PyTorch FloatTensor) - Souce positive hidden representations.
            * **x_n** (PyTorch FloatTensor) - Souce negative hidden representations.
            * **x_pt** (PyTorch FloatTensor, optional) - Target positive hidden representations. Default: None.
            * **x_nt** (PyTorch FloatTensor, optional) - Target negative hidden representations. Default: None.
            * **A_pt** (PyTorch FloatTensor or PyTorch sparse_coo_tensor, optional) - Transpose of column-normalized 
                positive part of the adjacency matrix. Default: None.
            * **A_nt** (PyTorch FloatTensor or PyTorch sparse_coo_tensor, optional) - Transpose of column-normalized 
                negative part of the adjacency matrix. Default: None.

        Return types:
            * **feat** (PyTorch FloatTensor) - Embedding matrix, with shape (num_nodes, 2*input_dim) for undirected graphs 
                and (num_nodes, 4*input_dim) for directed graphs.
        """
        if self._undirected:
            feat_p = self._w_p[0] * x_p
            feat_n = torch.zeros_like(feat_p)
            curr_p = x_p.clone()
            curr_n_aux = x_n.clone()
            j = 0
            for h in range(0, self._hop_p):
                if h > 0:
                    curr_p = torch.matmul(A_p, curr_p)
                    curr_n_aux = torch.matmul(A_p, curr_n_aux)
                    feat_p += self._w_p[h] * curr_p
                if h != self._hop_p - 1:
                    curr_n = torch.matmul(A_n, curr_n_aux)
                    feat_n += self._w_n[j] * curr_n
                    j += 1
                    for _ in range(self._hop_p - 2 - h):
                        curr_n = torch.matmul(A_p, curr_n)
                        feat_n += self._w_n[j] * curr_n
                        j += 1
            feat = torch.cat([feat_p, feat_n], dim=1)
        else:
            A_sp = A_p
            A_sn = A_n
            A_tp = A_pt
            A_tn = A_nt
            x_sp = x_p
            x_sn = x_n
            feat_sp = self._w_sp[0] * x_sp
            feat_sn = torch.zeros_like(feat_sp)
            feat_tp = self._w_tp[0] * x_pt
            feat_tn = torch.zeros_like(feat_tp)
            curr_sp = x_sp.clone()
            curr_sn_aux = x_sn.clone()
            curr_tp = x_pt.clone()
            curr_tn_aux = x_nt.clone()
            j = 0
            for h in range(0, self._hop_p):
                if h > 0:
                    curr_sp = torch.matmul(A_sp, curr_sp)
                    curr_sn_aux = torch.matmul(A_sp, curr_sn_aux)
                    curr_tp = torch.matmul(A_tp, curr_tp)
                    curr_tn_aux = torch.matmul(A_tp, curr_tn_aux)
                    feat_sp += self._w_sp[h] * curr_sp
                    feat_tp += self._w_tp[h] * curr_tp
                if h != self._hop_p - 1:
                    curr_sn = torch.matmul(A_sn, curr_sn_aux)
                    curr_tn = torch.matmul(A_tn, curr_tn_aux)
                    feat_sn += self._w_sn[j] * curr_sn
                    feat_tn += self._w_tn[j] * curr_tn
                    j += 1
                    for _ in range(self._hop_p - 2 - h):
                        curr_sn = torch.matmul(A_sp, curr_sn)
                        curr_tn = torch.matmul(A_tp, curr_tn)
                        feat_sn += self._w_sn[j] * curr_sn
                        feat_tn += self._w_tn[j] * curr_tn
                        j += 1
            feat = torch.cat([feat_sp, feat_sn, feat_tp, feat_tn], dim=1)
        return feat


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hop': 4}]
