import math
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F


class Attention(nn.Module):
    """Attention layer - Custom layer to perform weighted average over the second axis (axis=1)
        Transforming a tensor of size [N, W, H] to [N, 1, H].
        N: batch size
        W: number of words, different sentence length will need to be padded to have the same size for each mini-batch
        H: hidden state dimension or word embedding dimension
    Args:
        dim: The dimension of the word embedding
    Attributes:
        w: learnable weight matrix of size [dim, dim]
        v: learnable weight vector of size [dim]
    Examples::
        >>> m = models_pytorch.Attention(300)
        >>> input = Variable(torch.randn(4, 128, 300))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.att_weights = None
        self.w = nn.Parameter(torch.Tensor(dim, dim))
        self.v = nn.Parameter(torch.Tensor(dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, input):
        wplus = torch.mm(input.contiguous().view(-1, input.size()[2]), self.w)
        wplus = wplus.contiguous().view(-1, input.size()[1], self.w.size()[1])
        wplus = torch.tanh(wplus)
        att_w = torch.mm(wplus.contiguous().view(-1, wplus.size()[2]), self
            .v.contiguous().view(self.v.size()[0], 1))
        att_w = att_w.contiguous().view(-1, wplus.size()[1])
        att_w = F.softmax(att_w, dim=1)
        self.att_weights = att_w
        after_attention = torch.bmm(att_w.unsqueeze(1), input)
        return after_attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + '1' + ', ' + str(self.dim
            ) + ')'


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
