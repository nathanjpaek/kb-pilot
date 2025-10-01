import math
import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, emb_size, n_heads=8, mask=False):
        """
        Arguments:

        emb_size: Size of input Embeddings
        n_heads: Number of heads for MultiHead Attention

        Layers:

        tokeys: For Keys
        toquery: For Query
        tovalue: For Value

        combine_heads: Convert the ( (number of heads) * (emb_size) ) into (emb_size) for output

        Note:
            The input dimension we give to the Linear Layer (nn.Linear) is the last dimension (no. of columns) in size
            (batch_size ,seq_length, input_dims) ==> input_dims : give to nn.Linear
            So in order for the tensor to pass through linear layer having the last dim equal to input_dim of
            Linear layer is necessary.

        """
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.n_heads = n_heads
        """
        Explanation Comment:
            For a single head attention the input will be of emb_size and output will be of emb_size Linear(emb_size,emb_size) but in multihead attention
            there are number of heads (n_heads) and each head has it's own key(k), query(q) and value(v) so total number of k,q,v
            will be equal to n_heads.
            We can do that in a single linear layer by giving the size of output layer = n_head * emb_size (here emb_size = output size)
            Same for k,q,v. nn.Linear(emb_size,n_heads*emb_size)

        """
        self.tokeys = nn.Linear(emb_size, n_heads * emb_size, bias=False)
        self.toquery = nn.Linear(emb_size, n_heads * emb_size, bias=False)
        self.tovalue = nn.Linear(emb_size, n_heads * emb_size, bias=False)
        self.combine_heads = nn.Linear(n_heads * emb_size, emb_size)
        self.Attnweights = None
        self.mask = mask

    def __MatrixMask(self, matrix, mask_value=float('-inf'), diagonal=False):
        """
        In self attention mask is used when we are trying to predict the next work based on the previous sequence
        but attention inherently contains information about all the words in the sequence.
        In order to accuretly predict the next word without sort of cheating by looking ahead with attention we set all
        the attention weights of the next words, tokens = -inf or zero, so that there is no input from the sequence ahead
        of the current word


        [ 1 2 3                 [ 1 0 0
          4 5 6    =========>     4 5 0
          7 8 9 ]                 7 8 9 ]

        """
        row, column = matrix.size(-2), matrix.size(-1)
        offset = 0 if diagonal else 1
        upper_triangular_idx = torch.triu_indices(row, column, offset)
        matrix[..., upper_triangular_idx[0], upper_triangular_idx[1]
            ] = mask_value
        return matrix

    def forward(self, x):
        """
        Argument:

        x: shoud be of shape [batch,sequence length, emb_size]

        Local variable:

        b: batch size
        t: length of sequence
        e: embedding size of each individual input token/word
        h: number of heads

        """
        b, t, e = x.size()
        h = self.n_heads
        assert e == self.emb_size, f'Input size expected to be {self.emb_size} but got {e}'
        keys = self.tokeys(x).view(b, t, h, e)
        query = self.toquery(x).view(b, t, h, e)
        value = self.tovalue(x).view(b, t, h, e)
        """
        Pytorch Contiguous() function is used to keep the indexes which are side by side, side by side in memory also

        """
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        query = query.transpose(1, 2).contiguous().view(b * h, t, e)
        value = value.transpose(1, 2).contiguous().view(b * h, t, e)
        raw_weights = torch.bmm(query, keys.transpose(1, 2))
        raw_weights /= math.sqrt(e)
        assert raw_weights.size() == (b * h, t, t
            ), f'expected shape {b * h, t, t} got {raw_weights.size()}'
        if self.mask:
            raw_weights = self.__MatrixMask(raw_weights)
        weights = F.softmax(raw_weights, dim=2)
        self.Attnweights = weights
        out = torch.bmm(weights, value).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        out = self.combine_heads(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'emb_size': 4}]
