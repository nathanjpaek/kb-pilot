import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def orthogonal_matrix_chunk(cols, qr_uniform_q=False, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), 'reduced')
    q, r = map(lambda t: t, (q, r))
    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0,
    qr_uniform_q=False, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []
    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q,
            device=device)
        block_list.append(q)
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q,
            device=device)
        block_list.append(q[:remaining_rows])
    final_matrix = torch.cat(block_list)
    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim
            =1)
    elif scaling == 1:
        multiplier = math.sqrt(float(nb_columns)) * torch.ones((nb_rows,),
            device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')
    return torch.diag(multiplier) @ final_matrix


def generalized_kernel(data, *, projection_matrix, kernel_fn=nn.ReLU(
    inplace=True), kernel_epsilon=0.001, normalize_data=True, device=None):
    _b, _h, *_ = data.shape
    data_normalizer = data.shape[-1] ** -0.25 if normalize_data else 1.0
    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon
    data = data_normalizer * data
    data = torch.matmul(data, projection_matrix.T)
    data = kernel_fn(data) + kernel_epsilon
    return data.type_as(data)


def linear_attention(q, k, v):
    L = k.shape[-2]
    D_inv = 1.0 / torch.einsum('...nd,...d->...n', q, k.mean(dim=-2))
    context = torch.einsum('...nd,...ne->...de', k / float(L), v)
    del k, v
    out = torch.einsum('...n,...nd->...nd', D_inv, q)
    del D_inv, q
    out = torch.einsum('...nd,...de->...ne', out, context)
    return out


def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=
    True, eps=0.0001, device=None):
    b, h, *_ = data.shape
    data_normalizer = data.shape[-1] ** -0.25 if normalize_data else 1.0
    ratio = projection_matrix.shape[0] ** -0.5
    projection = projection_matrix.unsqueeze(0).repeat(h, 1, 1)
    projection = projection.unsqueeze(0).repeat(b, 1, 1, 1)
    projection = projection.type_as(data)
    data_dash = torch.einsum('...id,...jd->...ij', data_normalizer * data,
        projection)
    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = diag_data / 2.0 * data_normalizer ** 2
    diag_data = diag_data.unsqueeze(dim=-1)
    if is_query:
        data_dash = ratio * (torch.exp(data_dash - diag_data - torch.max(
            data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (torch.exp(data_dash - diag_data - torch.max(
            data_dash)) + eps)
    return data_dash.type_as(data)


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module,
        type)]


def get_module_device(module):
    return next(module.parameters()).device


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-05):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps
            )
        x = self.a_2 * (x - mean)
        x /= std
        x += self.b_2
        return x


class MultiheadAttention(nn.Module):

    def __init__(self, d_model, heads, k_dim=None, v_dim=None, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        if k_dim is None:
            k_dim = d_model
        if v_dim is None:
            v_dim = d_model
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(k_dim, d_model)
        self.to_value = nn.Linear(v_dim, d_model)
        self.to_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch, L = query.shape[:2]
        q = self.to_query(query).view(batch, L, self.heads, self.d_k).permute(
            0, 2, 1, 3)
        k = self.to_key(key).view(batch, L, self.heads, self.d_k).permute(0,
            2, 1, 3)
        v = self.to_value(value).view(batch, L, self.heads, self.d_k).permute(
            0, 2, 1, 3)
        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, L, -1)
        out = self.to_out(out)
        return out


class FastAttention(nn.Module):

    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0,
        generalized_attention=False, kernel_fn=nn.ReLU(inplace=True),
        qr_uniform_q=False, no_projection=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads))
            )
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        if not no_projection:
            self.create_projection = partial(gaussian_orthogonal_random_matrix,
                nb_rows=self.nb_features, nb_columns=dim_heads, scaling=
                ortho_scaling, qr_uniform_q=qr_uniform_q)
            projection_matrix = self.create_projection()
            self.register_buffer('projection_matrix', projection_matrix)
        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.no_projection = no_projection

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device
        if self.no_projection:
            q = q.softmax(dim=-1)
            k.softmax(dim=-2)
        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn=self.
                kernel_fn, projection_matrix=self.projection_matrix, device
                =device)
            q, k = map(create_kernel, (q, k))
        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.
                projection_matrix, device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)
        attn_fn = linear_attention
        out = attn_fn(q, k, v)
        return out


class SelfAttention(nn.Module):

    def __init__(self, dim, k_dim=None, heads=8, local_heads=0,
        local_window_size=256, nb_features=None, feature_redraw_interval=
        1000, generalized_attention=False, kernel_fn=nn.ReLU(inplace=True),
        qr_uniform_q=False, dropout=0.0, no_projection=False):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = dim // heads
        inner_dim = dim_head * heads
        if k_dim is None:
            k_dim = dim
        self.fast_attention = FastAttention(dim_head, nb_features,
            generalized_attention=generalized_attention, kernel_fn=
            kernel_fn, qr_uniform_q=qr_uniform_q, no_projection=no_projection)
        self.heads = heads
        self.dim = dim
        self.to_query = nn.Linear(dim, inner_dim)
        self.to_key = nn.Linear(k_dim, inner_dim)
        self.to_value = nn.Linear(k_dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))
        self.max_tokens = 2 ** 16

    def check_redraw_projections(self):
        if not self.training:
            return
        if exists(self.feature_redraw_interval
            ) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(self)
            fast_attentions = find_modules(self, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)
            self.calls_since_last_redraw.zero_()
            return
        self.calls_since_last_redraw += 1

    def _batched_forward(self, q, k, v):
        b1, h, n1 = q.shape[:3]
        out = torch.empty((b1, h, n1, self.dim // h), dtype=q.dtype, device
            =q.device)
        shift = self.max_tokens // n1
        for i_b in range(0, b1, shift):
            start = i_b
            end = min(i_b + shift, b1)
            out[start:end] = self.fast_attention(q[start:end], k[start:end],
                v[start:end])
        return out

    def forward(self, query, key, value, **kwargs):
        self.check_redraw_projections()
        b1, n1, _, h = *query.shape, self.heads
        b2, n2, _, h = *key.shape, self.heads
        q = self.to_query(query)
        k = self.to_key(key)
        v = self.to_value(value)
        q = q.reshape(b1, n1, h, -1).permute(0, 2, 1, 3)
        k = k.reshape(b2, n2, h, -1).permute(0, 2, 1, 3)
        v = v.reshape(b2, n2, h, -1).permute(0, 2, 1, 3)
        if b1 * n1 > self.max_tokens or b2 * n2 > self.max_tokens:
            out = self._batched_forward(q, k, v)
        else:
            out = self.fast_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(b1, n1, -1)
        out = self.to_out(out)
        return self.dropout(out)


class SequenceWeight(nn.Module):

    def __init__(self, d_model, heads, dropout=0.1):
        super(SequenceWeight, self).__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, msa):
        B, N, L = msa.shape[:3]
        msa = msa.permute(0, 2, 1, 3)
        tar_seq = msa[:, :, 0].unsqueeze(2)
        q = self.to_query(tar_seq).view(B, L, 1, self.heads, self.d_k).permute(
            0, 1, 3, 2, 4).contiguous()
        k = self.to_key(msa).view(B, L, N, self.heads, self.d_k).permute(0,
            1, 3, 4, 2).contiguous()
        q = q * self.scale
        attn = torch.matmul(q, k)
        attn = F.softmax(attn, dim=-1)
        return self.dropout(attn)


class FeedForwardLayer(nn.Module):

    def __init__(self, d_model, d_ff, p_drop=0.1):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p_drop, inplace=True)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, src):
        src = self.linear2(self.dropout(F.relu_(self.linear1(src))))
        return src


class TiedMultiheadAttention(nn.Module):

    def __init__(self, d_model, heads, k_dim=None, v_dim=None, dropout=0.1):
        super(TiedMultiheadAttention, self).__init__()
        if k_dim is None:
            k_dim = d_model
        if v_dim is None:
            v_dim = d_model
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scaling = 1 / math.sqrt(self.d_k)
        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(k_dim, d_model)
        self.to_value = nn.Linear(v_dim, d_model)
        self.to_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, query, key, value, return_att=False):
        B, N, L = query.shape[:3]
        q = self.to_query(query).view(B, N, L, self.heads, self.d_k).permute(
            0, 1, 3, 2, 4).contiguous()
        k = self.to_key(key).view(B, N, L, self.heads, self.d_k).permute(0,
            1, 3, 4, 2).contiguous()
        v = self.to_value(value).view(B, N, L, self.heads, self.d_k).permute(
            0, 1, 3, 2, 4).contiguous()
        scale = self.scaling / math.sqrt(N)
        q = q * scale
        attention = torch.einsum('bnhik,bnhkj->bhij', q, k)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        attention = attention.unsqueeze(1)
        out = torch.matmul(attention, v)
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(B, N, L, -1)
        out = self.to_out(out)
        if return_att:
            attention = attention.squeeze(1)
            attention = 0.5 * (attention + attention.permute(0, 1, 3, 2))
            attention = attention.permute(0, 3, 1, 2)
            return out, attention
        return out


class SoftTiedMultiheadAttention(nn.Module):

    def __init__(self, d_model, heads, k_dim=None, v_dim=None, dropout=0.1):
        super(SoftTiedMultiheadAttention, self).__init__()
        if k_dim is None:
            k_dim = d_model
        if v_dim is None:
            v_dim = d_model
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.seq_weight = SequenceWeight(d_model, heads, dropout=dropout)
        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(k_dim, d_model)
        self.to_value = nn.Linear(v_dim, d_model)
        self.to_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, query, key, value, return_att=False):
        B, N, L = query.shape[:3]
        seq_weight = self.seq_weight(query)
        seq_weight = seq_weight.permute(0, 4, 2, 1, 3)
        q = self.to_query(query).view(B, N, L, self.heads, self.d_k).permute(
            0, 1, 3, 2, 4).contiguous()
        k = self.to_key(key).view(B, N, L, self.heads, self.d_k).permute(0,
            1, 3, 4, 2).contiguous()
        v = self.to_value(value).view(B, N, L, self.heads, self.d_k).permute(
            0, 1, 3, 2, 4).contiguous()
        q = q * seq_weight
        k = k * self.scale
        attention = torch.einsum('bnhik,bnhkj->bhij', q, k)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        attention = attention
        del q, k, seq_weight
        out = torch.einsum('bhij,bnhjk->bnhik', attention, v)
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(B, N, L, -1)
        out = self.to_out(out)
        if return_att:
            attention = attention.squeeze(1)
            attention = 0.5 * (attention + attention.permute(0, 1, 3, 2))
            attention = attention.permute(0, 2, 3, 1)
            return out, attention
        return out


class AxialEncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, heads, p_drop=0.1, performer_opts=
        None, use_tied_row=False, use_tied_col=False, use_soft_row=False):
        super(AxialEncoderLayer, self).__init__()
        self.use_performer = performer_opts is not None
        self.use_tied_row = use_tied_row
        self.use_tied_col = use_tied_col
        self.use_soft_row = use_soft_row
        if use_tied_row:
            self.attn_L = TiedMultiheadAttention(d_model, heads, dropout=p_drop
                )
        elif use_soft_row:
            self.attn_L = SoftTiedMultiheadAttention(d_model, heads,
                dropout=p_drop)
        elif self.use_performer:
            self.attn_L = SelfAttention(dim=d_model, heads=heads, dropout=
                p_drop, generalized_attention=True, **performer_opts)
        else:
            self.attn_L = MultiheadAttention(d_model, heads, dropout=p_drop)
        if use_tied_col:
            self.attn_N = TiedMultiheadAttention(d_model, heads, dropout=p_drop
                )
        elif self.use_performer:
            self.attn_N = SelfAttention(dim=d_model, heads=heads, dropout=
                p_drop, generalized_attention=True, **performer_opts)
        else:
            self.attn_N = MultiheadAttention(d_model, heads, dropout=p_drop)
        self.ff = FeedForwardLayer(d_model, d_ff, p_drop=p_drop)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop, inplace=True)
        self.dropout2 = nn.Dropout(p_drop, inplace=True)
        self.dropout3 = nn.Dropout(p_drop, inplace=True)

    def forward(self, src, return_att=False):
        B, N, L = src.shape[:3]
        src2 = self.norm1(src)
        if self.use_tied_row or self.use_soft_row:
            src2 = self.attn_L(src2, src2, src2)
        else:
            src2 = src2.reshape(B * N, L, -1)
            src2 = self.attn_L(src2, src2, src2)
            src2 = src2.reshape(B, N, L, -1)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        if self.use_tied_col:
            src2 = src2.permute(0, 2, 1, 3)
            src2 = self.attn_N(src2, src2, src2)
            src2 = src2.permute(0, 2, 1, 3)
        else:
            src2 = src2.permute(0, 2, 1, 3).reshape(B * L, N, -1)
            src2 = self.attn_N(src2, src2, src2)
            src2 = src2.reshape(B, L, N, -1).permute(0, 2, 1, 3)
        src = src + self.dropout2(src2)
        src2 = self.norm3(src)
        src2 = self.ff(src2)
        src = src + self.dropout3(src2)
        return src


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_ff': 4, 'heads': 4}]
