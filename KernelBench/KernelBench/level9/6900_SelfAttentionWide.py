import torch
from torch import nn
import torch.nn.functional as F


def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false

    In place operation

    :param tns:
    :return:
    """
    _b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval


class SelfAttentionWide(nn.Module):

    def __init__(self, emb, heads=8, mask=False):
        """

        :param emb:
        :param heads:
        :param mask:
        """
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.mask = mask
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)
        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):
        """ @kewlcoder -
        Here,
        b: denotes the batch size
        t: denotes the max sequence length(max number of words/tokens in the sentence/input)
        e: the embedding dimensionality
        """
        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'
        """ @kewlcoder -
        x(input) when fed to a linear layer (b,t,e) * (e, e*h) => (b,t,e*h)
        """
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)
        """
        Since the head and batch dimension are not next to each other, we need to transpose before we reshape.
        (This is costly, but it seems to be unavoidable.)
        """
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries / e ** (1 / 4)
        keys = keys / e ** (1 / 4)
        """ @kewlcoder - we divide by sqrt(e) because in a 2D vector space if a vector has c value in each dimension, the 
        aggregate vector becomes sqrt(2) * c. Thus, for n-dim vector space it would have an impact of sqrt(n). Thus, as 
        dim(e) increases, the product would become bigger and bigger and to supress that, we divide by sqrt(e).
        For every item in the batch and for every head individually, compute dot = (Q*K/sqrt(emb))
        """
        dot = torch.bmm(queries, keys.transpose(1, 2))
        """ @kewlcoder - Here, each element (i,j) of the sub-matrix (t,t) represents the attention weight to be given to word j for 
        calculating the weighted attention generated vector for word i in the sequence(or vice-versa).
        """
        assert dot.size() == (b * h, t, t)
        if self.mask:
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t, e)
        """
        can also use - 
        https://pytorch.org/docs/stable/generated/torch.einsum.html
        https://github.com/pbloem/former/issues/4
        """
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        """ @kewlcoder - (b,t,h*e)(h*e, e) => (b,t,e) -> We finally get attention weighted 
        embedding/hidden layer that can be used for various downstream tasks.
        """
        return self.unifyheads(out)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'emb': 4}]
