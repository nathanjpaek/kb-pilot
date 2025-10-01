from torch.nn import Module
import torch
from torch.nn.functional import cosine_similarity


def triplet_margin_cosine_loss(anchor, positive, negative, margin=1.0, eps=
    1e-08, sum_loss=False):
    'Creates a criterion that measures the triplet cosine loss given input\n    tensors x1, x2, x3 and a margin with a value greater than 0.\n    This is used for measuring a relative similarity between samples. A triplet\n    is composed by `a`, `p` and `n`: anchor, positive example and negative\n    example(s) respectively. The shape of the anchor and positive variables should \n    be\n    math:`(N, D)`.\n    The shape of the negative variable should be\n    math:`(N, D)`, for 1 negative sample, or\n    math:`(N, m, D)`, for m negative samples.\n\n\n    .. math::\n        L(a, p, n) = \x0crac{1}{N} \\left( \\sum_{i=1}^N \\max \\{0, margin - cos(a_i, p_i) + cos(a_i, n_i)\\} \right)\n\n    Args:\n        anchor: anchor input tensor\n        positive: positive input tensor\n        negative: negative input tensor\n        margin: the margin value. Default: 1\n        eps: small epsilon value to avoid numerical issues. Default: 1e-6\n        sum_loss: if True the hinge loss will be summed across batch instances\n\n    Shape:\n        - Input: :math:`(N, D)` where `D = vector dimension`\n        - Output: :math:`(N, 1)`\n\n    Example::\n\n        >>> input1 = autograd.Variable(torch.randn(100, 128))\n        >>> input2 = autograd.Variable(torch.randn(100, 128))\n        >>> input3 = autograd.Variable(torch.randn(100, 10, 128))\n        >>> output = triplet_margin_cosine_loss(input1, input2, input3)\n        >>> output.backward()\n    '
    assert anchor.size() == positive.size(
        ), 'Input sizes between anchor and positive must be equal.'
    assert anchor.dim() == 2, 'Anchor and positive must be 2D matrices.'
    assert negative.dim(
        ) <= 3, 'Negative must be 2D (1 negative sample) or 3D matrix (multiple negatives).'
    assert margin > 0.0, 'Margin should be positive value.'
    if negative.dim() == 2:
        assert anchor.size() == negative.size(
            ), 'Input sizes between anchor and negative must be equal (if 1 negative sample).'
        d_p = cosine_similarity(anchor, positive, eps=eps)
        d_n = cosine_similarity(anchor, negative, eps=eps)
        dist_hinge = torch.clamp(margin - d_p + d_n, min=0.0)
    else:
        assert anchor.size()[0] == negative.size()[0] and anchor.size()[1
            ] == negative.size()[2
            ], 'If using multiple negatives samples, their size: [B, #neg, emb_size].'
        d_p = cosine_similarity(anchor, positive, eps=eps)
        d_n = cosine_similarity(anchor.unsqueeze(1), negative, dim=2, eps=eps)
        dist_hinge = torch.clamp(margin - d_p.unsqueeze(1) + d_n, min=0.0).sum(
            dim=1)
    if not sum_loss:
        loss = torch.mean(dist_hinge)
    else:
        loss = torch.sum(dist_hinge)
    return loss


class TripletMarginCosineLoss(Module):
    'Creates a criterion that measures the triplet cosine loss given an input\n    tensors x1, x2, x3 and a margin with a value greater than 0.\n    This is used for measuring a relative similarity between samples. A triplet\n    is composed by `a`, `p` and `n`: anchor, positive examples and negative\n    example(s) respectively. The shape of the anchor and positive variables should \n    be\n    math:`(N, D)`.\n    The shape of the negative variable should be\n    math:`(N, D)`, for 1 negative sample, or\n    math:`(N, m, D)`, for m negative samples.\n\n\n    .. math::\n        L(a, p, n) = \x0crac{1}{N} \\left( \\sum_{i=1}^N \\max \\{0, margin - cos(a_i, p_i) + cos(a_i, n_i)\\} \right)\n\n    Args:\n        anchor: anchor input tensor\n        positive: positive input tensor\n        negative: negative input tensor\n\n    Shape:\n        - Input: :math:`(N, D)` where `D = vector dimension`\n        - Output: :math:`(N, 1)`\n\n    >>> triplet_loss = nn.TripletMarginCosineLoss(margin=1.0, p=2)\n    >>> input1 = autograd.Variable(torch.randn(100, 128))\n    >>> input2 = autograd.Variable(torch.randn(100, 128))\n    >>> input3 = autograd.Variable(torch.randn(100, 10, 128))\n    >>> output = triplet_loss(input1, input2, input3)\n    >>> output.backward()\n    '

    def __init__(self, margin=1.0, eps=1e-08, sum_loss=False):
        super(TripletMarginCosineLoss, self).__init__()
        self.margin = margin
        self.eps = eps
        self.sum_loss = sum_loss

    def forward(self, anchor, positive, negative):
        return triplet_margin_cosine_loss(anchor, positive, negative, self.
            margin, self.eps, self.sum_loss)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
