import torch
from torch import nn
from torch.nn import functional as F


def same_tensor(tensor, *args):
    """ Do the input tensors all point to the same underlying data """
    for other in args:
        if not torch.is_tensor(other):
            return False
        if tensor.device != other.device:
            return False
        if tensor.dtype != other.dtype:
            return False
        if tensor.data_ptr() != other.data_ptr():
            return False
    return True


class MultiHeadedAttention(nn.Module):
    """ Implement a multi-headed attention module """

    def __init__(self, embed_dim, num_heads=1):
        """ Initialize the attention module """
        super(MultiHeadedAttention, self).__init__()
        assert embed_dim % num_heads == 0, f'num_heads={num_heads} should evenly divide embed_dim={embed_dim}'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.scale = self.projection_dim ** -0.5
        self.input_weights = nn.Parameter(torch.Tensor(3 * embed_dim,
            embed_dim))
        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """ Reset parameters using xavier initialization """
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(self.input_weights, gain)
        nn.init.xavier_uniform_(self.output_projection.weight, gain)

    def project(self, inputs, index=0, chunks=1):
        """ Produce a linear projection using the weights """
        batch_size = inputs.shape[0]
        start = index * self.embed_dim
        end = start + chunks * self.embed_dim
        projections = F.linear(inputs, self.input_weights[start:end]).chunk(
            chunks, dim=-1)
        output_projections = []
        for projection in projections:
            output_projections.append(projection.view(batch_size, -1, self.
                num_heads, self.projection_dim).transpose(2, 1).contiguous(
                ).view(batch_size * self.num_heads, -1, self.projection_dim))
        return output_projections

    def attention(self, values, keys, queries, key_mask=None, mask=None):
        """ Scaled dot product attention with optional masks """
        logits = self.scale * torch.bmm(queries, keys.transpose(2, 1))
        if mask is not None:
            logits += mask
        if key_mask is not None:
            logits_shape = logits.shape
            batch_size = logits_shape[0] // self.num_heads
            logits = logits.view(batch_size, self.num_heads, logits_shape[1
                ], logits_shape[2])
            logits.masked_fill_(key_mask[:, None, None], float('-inf'))
            logits = logits.view(logits_shape)
        attn_weights = F.softmax(logits, dim=-1)
        attended = torch.bmm(attn_weights, values)
        batch_size = queries.shape[0] // self.num_heads
        return attended.view(batch_size, self.num_heads, -1, self.
            projection_dim).transpose(2, 1).contiguous().view(batch_size, -
            1, self.num_heads * self.projection_dim)

    def forward(self, values, keys, queries, key_mask=None, attention_mask=
        None, num_queries=0):
        """ Forward pass of the attention """
        None
        None
        None
        if same_tensor(values, keys, queries):
            values, keys, queries = self.project(values, chunks=3)
        elif same_tensor(values, keys):
            values, keys = self.project(values, chunks=2)
            queries, = self.project(queries, 2)
        else:
            values, = self.project(values, 0)
            keys, = self.project(keys, 1)
            queries, = self.project(queries, 2)
        if num_queries:
            queries = queries[:, -num_queries:]
        attended = self.attention(values, keys, queries, key_mask,
            attention_mask)
        return self.output_projection(attended)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'embed_dim': 4}]
