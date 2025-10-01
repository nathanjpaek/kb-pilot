import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import *
import torch.optim.lr_scheduler
import torch.quantization
from torch.nn.parameter import Parameter
import torch.onnx
import torch.testing


class EltwiseAdd(nn.Module):

    def __init__(self, inplace=False):
        """Element-wise addition"""
        super().__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res += t
        else:
            for t in input[1:]:
                res = res + t
        return res


class EltwiseMult(nn.Module):

    def __init__(self, inplace=False):
        """Element-wise multiplication"""
        super().__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res *= t
        else:
            for t in input[1:]:
                res = res * t
        return res


class Matmul(nn.Module):
    """
    A wrapper module for matmul operation between 2 tensors.
    """

    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, a: 'torch.Tensor', b: 'torch.Tensor'):
        return a.matmul(b)


class BatchMatmul(nn.Module):
    """
    A wrapper module for torch.bmm operation between 2 tensors.
    """

    def __init__(self):
        super(BatchMatmul, self).__init__()

    def forward(self, a: 'torch.Tensor', b: 'torch.Tensor'):
        return torch.bmm(a, b)


class BahdanauAttention(nn.Module):
    """
    It should be very similar to tf.contrib.seq2seq.BahdanauAttention
    """

    def __init__(self, query_size, key_size, num_units, normalize=False,
        dropout=0, batch_first=False):
        super(BahdanauAttention, self).__init__()
        self.normalize = normalize
        self.batch_first = batch_first
        self.num_units = num_units
        self.linear_q = nn.Linear(query_size, num_units, bias=False)
        self.linear_k = nn.Linear(key_size, num_units, bias=False)
        self.linear_att = Parameter(torch.Tensor(num_units))
        self.dropout = nn.Dropout(dropout)
        self.mask = None
        self.eltwiseadd_qk = EltwiseAdd()
        self.eltwiseadd_norm_bias = EltwiseAdd()
        self.eltwisemul_norm_scaler = EltwiseMult()
        self.tanh = nn.Tanh()
        self.matmul_score = Matmul()
        self.softmax_att = nn.Softmax(dim=-1)
        self.context_matmul = BatchMatmul()
        if self.normalize:
            self.normalize_scalar = Parameter(torch.Tensor(1))
            self.normalize_bias = Parameter(torch.Tensor(num_units))
        else:
            self.register_parameter('normalize_scalar', None)
            self.register_parameter('normalize_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.num_units)
        self.linear_att.data.uniform_(-stdv, stdv)
        if self.normalize:
            self.normalize_scalar.data.fill_(stdv)
            self.normalize_bias.data.zero_()

    def set_mask(self, context_len, context):
        """
        sets self.mask which is applied before softmax
        ones for inactive context fields, zeros for active context fields

        :param context_len: b
        :param context: if batch_first: (b x t_k x n) else: (t_k x b x n)

        self.mask: (b x t_k)
        """
        if self.batch_first:
            max_len = context.size(1)
        else:
            max_len = context.size(0)
        indices = torch.arange(0, max_len, dtype=torch.int64, device=
            context.device)
        self.mask = indices >= context_len.unsqueeze(1)

    def calc_score(self, att_query, att_keys):
        """
        Calculate Bahdanau score

        :param att_query: b x t_q x n
        :param att_keys: b x t_k x n

        return b x t_q x t_k scores
        """
        b, t_k, n = att_keys.size()
        t_q = att_query.size(1)
        att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n)
        att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n)
        sum_qk = self.eltwiseadd_qk(att_query, att_keys)
        if self.normalize:
            sum_qk = self.eltwiseadd_norm_bias(sum_qk, self.normalize_bias)
            tmp = self.linear_att
            linear_att = tmp / tmp.norm()
            linear_att = linear_att
            linear_att = self.eltwisemul_norm_scaler(linear_att, self.
                normalize_scalar)
        else:
            linear_att = self.linear_att
        out = self.matmul_score(self.tanh(sum_qk), linear_att)
        return out

    def forward(self, query, keys):
        """

        :param query: if batch_first: (b x t_q x n) else: (t_q x b x n)
        :param keys: if batch_first: (b x t_k x n) else (t_k x b x n)

        :returns: (context, scores_normalized)
        context: if batch_first: (b x t_q x n) else (t_q x b x n)
        scores_normalized: if batch_first (b x t_q x t_k) else (t_q x b x t_k)
        """
        if not self.batch_first:
            keys = keys.transpose(0, 1)
            if query.dim() == 3:
                query = query.transpose(0, 1)
        if query.dim() == 2:
            single_query = True
            query = query.unsqueeze(1)
        else:
            single_query = False
        b = query.size(0)
        t_k = keys.size(1)
        t_q = query.size(1)
        processed_query = self.linear_q(query)
        processed_key = self.linear_k(keys)
        scores = self.calc_score(processed_query, processed_key)
        if self.mask is not None:
            mask = self.mask.unsqueeze(1).expand(b, t_q, t_k)
            scores.data.masked_fill_(mask, -65504.0)
        scores_normalized = self.softmax_att(scores)
        scores_normalized = self.dropout(scores_normalized)
        context = self.context_matmul(scores_normalized, keys)
        if single_query:
            context = context.squeeze(1)
            scores_normalized = scores_normalized.squeeze(1)
        elif not self.batch_first:
            context = context.transpose(0, 1)
            scores_normalized = scores_normalized.transpose(0, 1)
        return context, scores_normalized


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'query_size': 4, 'key_size': 4, 'num_units': 4}]
