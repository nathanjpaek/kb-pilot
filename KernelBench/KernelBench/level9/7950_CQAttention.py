import torch
import torch.nn.parallel
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn


def mask_logits(inputs, mask, mask_value=-1e+30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


class Conv1D(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0,
        bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim,
            kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        return x.transpose(1, 2)


class CQAttention(nn.Module):

    def __init__(self, dim, drop_rate=0.0):
        super(CQAttention, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1,
            stride=1, padding=0, bias=True)

    def forward(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(context, query)
        score_ = nn.Softmax(dim=2)(mask_logits(score, q_mask.unsqueeze(1)))
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))
        score_t = score_t.transpose(1, 2)
        c2q = torch.matmul(score_, query)
        q2c = torch.matmul(torch.matmul(score_, score_t), context)
        output = torch.cat([context, c2q, torch.mul(context, c2q), torch.
            mul(context, q2c)], dim=2)
        output = self.cqa_linear(output)
        return output

    def trilinear_attention(self, context, query):
        _batch_size, c_seq_len, _dim = context.shape
        _batch_size, q_seq_len, _dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand([-1, -1, q_seq_len])
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand([-1,
            c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2
        return res


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4]
        ), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
