import torch
from torch import nn
import torch.nn.functional as F


class VectorQuantizeLayer_GB(nn.Module):

    def __init__(self, input_dim, vq_size, vq_dim, temp=(1.0, 0.1, 0.99),
        groups=1, combine_groups=True, time_first=True, activation=nn.GELU(
        ), weight_proj_depth=1, weight_proj_factor=1):
        """Vector quantization using gumbel softmax
        Args:
            input_dim: input dimension (channels)
            vq_size: number of quantized vectors per group
            vq_dim: dimensionality of the resulting quantized vector
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()
        self.input_dim = input_dim
        self.vq_size = vq_size
        self.groups = groups
        self.combine_groups = combine_groups
        self.time_first = time_first
        self.out_dim = vq_dim
        assert vq_dim % groups == 0, f'dim {vq_dim} must be divisible by groups {groups} for concatenation'
        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1
        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * vq_size,
            var_dim))
        nn.init.uniform_(self.vars)
        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                return nn.Sequential(nn.Linear(input_dim, output_dim),
                    activation)
            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(*[block(self.input_dim if i ==
                0 else inner_dim, inner_dim) for i in range(
                weight_proj_depth - 1)], nn.Linear(inner_dim, groups * vq_size)
                )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * vq_size)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)
        assert len(temp) == 3, temp
        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        self.curr_temp = max(self.max_temp * self.temp_decay ** num_updates,
            self.min_temp)

    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product
            p = [range(self.vq_size)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(inds, dtype=torch.long,
                device=self.vars.device).flatten()
            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(self.
                    vq_size ** self.groups, -1)
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.vq_size * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        indices = self.get_codebook_indices()
        return self.vars.squeeze(0).index_select(0, indices).view(self.
            vq_size ** self.groups, -1)

    def sample_from_codebook(self, b, n):
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.groups)
        cb_size = indices.size(0)
        assert n < cb_size, f'sample size {n} is greater than size of codebook {cb_size}'
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]
        z = self.vars.squeeze(0).index_select(0, indices.flatten()).view(b,
            n, -1)
        return z

    def to_codebook_index(self, indices):
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.groups):
            exponent = self.groups - i - 1
            res += indices[..., i] * self.vq_size ** exponent
        return res

    def forward(self, x, produce_targets=False):
        result = {'vq_size': self.vq_size * self.groups}
        if not self.time_first:
            x = x.transpose(1, 2)
        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        x = x.view(bsz * tsz * self.groups, -1)
        _, k = x.max(-1)
        hard_x = x.new_zeros(*x.shape).scatter_(-1, k.view(-1, 1), 1.0).view(
            bsz * tsz, self.groups, -1)
        result['temp'] = self.curr_temp
        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True
                ).type_as(x)
        else:
            x = hard_x
        x = x.view(bsz * tsz, -1)
        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)
        if produce_targets:
            result['targets'] = x.view(bsz * tsz * self.groups, -1).argmax(dim
                =-1).view(bsz, tsz, self.groups).detach()
        x = x.unsqueeze(-1) * vars
        x = x.view(bsz * tsz, self.groups, self.vq_size, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)
        if not self.time_first:
            x = x.transpose(1, 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'vq_size': 4, 'vq_dim': 4}]
