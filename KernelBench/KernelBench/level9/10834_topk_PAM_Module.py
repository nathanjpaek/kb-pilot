from torch.nn import Module
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.module import Module


def mask_softmax(input, mask=None, dim=-1):
    """Applies a softmax function.

        Softmax is defined as:

        :math:`\\text{Softmax}(x_{i}) = \\frac{exp(x_i)}{\\sum_j exp(x_j)}`

        It is applied to all slices along dim, and will re-scale them so that the elements
        lie in the range `(0, 1)` and sum to 1.

        See :class:`~torch.nn.Softmax` for more details.

        Arguments:
            input (Tensor): input
            dim (int): A dimension along which softmax will be computed.
            dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            If specified, the input tensor is casted to :attr:`dtype` before the operation
            is performed. This is useful for preventing data type overflows. Default: None.


        .. note::
            This function doesn't work directly with NLLLoss,
            which expects the Log to be computed between the Softmax and itself.
            Use log_softmax instead (it's faster and has better numerical properties).

        """
    if mask is None:
        return F.softmax(input, dim=dim, _stacklevel=5)
    else:
        max_input = input.max(dim=dim, keepdim=True)
        exp_input = torch.exp(input - max_input[0])
        mask_exp_input = torch.mul(exp_input, mask)
        sum_mask_exp_input = torch.sum(mask_exp_input, dim=dim, keepdim=True
            ) + 1e-10
        return torch.div(mask_exp_input, sum_mask_exp_input)


def mvmask_softmax(input, mask=None, dim=-1):
    """Applies a softmax function.

        Softmax is defined as:

        :math:`\\text{Softmax}(x_{i}) = \\frac{exp(x_i)}{\\sum_j exp(x_j)}`

        It is applied to all slices along dim, and will re-scale them so that the elements
        lie in the range `(0, 1)` and sum to 1.

        See :class:`~torch.nn.Softmax` for more details.

        Arguments:
            input (Tensor): input
            dim (int): A dimension along which softmax will be computed.
            dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            If specified, the input tensor is casted to :attr:`dtype` before the operation
            is performed. This is useful for preventing data type overflows. Default: None.


        .. note::
            This function doesn't work directly with NLLLoss,
            which expects the Log to be computed between the Softmax and itself.
            Use log_softmax instead (it's faster and has better numerical properties).

        """
    if mask is None:
        return F.softmax(input, dim=dim, _stacklevel=5)
    else:
        if torch.is_tensor(mask):
            return mask_softmax(input, mask=mask, dim=dim)
        mask = [mask[0], mask[1]]
        N, _H, _W = mask[0].size()
        max_input = input.max(dim=dim, keepdim=True)
        exp_input = torch.exp(input - max_input[0])
        if N == 1:
            mask_exp_input = torch.mul(exp_input, mask[0])
            sum_mask_exp_input = torch.sum(mask_exp_input, dim=dim, keepdim
                =True) + 1e-10
            return torch.div(mask_exp_input, sum_mask_exp_input)
        else:
            Sm = 0
            for i in range(N):
                mask_exp_input = torch.mul(exp_input, mask[0][i])
                sum_mask_exp_input = torch.sum(mask_exp_input, dim=dim,
                    keepdim=True) + 1e-10
                Sm = Sm + torch.div(mask_exp_input, sum_mask_exp_input)
            return torch.mul(Sm, mask[1])


class Mask_Softmax(Module):
    """Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1

    Softmax is defined as:

    .. math::
        \\text{Softmax}(x_{i}) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}

    Shape:
        - Input: any shape
        - Output: same as input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Arguments:
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).

    .. note::
        This module doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use `LogSoftmax` instead (it's faster and has better numerical properties).

    Examples::

        >>> m = nn.Softmax()
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """

    def __init__(self, dim=None):
        super(Mask_Softmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        return mvmask_softmax(input[0], input[1], self.dim)


class topk_PAM_Module(nn.Module):
    """ Position attention module"""

    def __init__(self, in_dim, key_dim, out_dim, topk=10):
        super(topk_PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.topk = topk
        self.key_channels = key_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=
            key_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim,
            kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=
            out_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = Mask_Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height
            ).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        proj_value = self.value_conv(x)
        proj_value = proj_value.view(m_batchsize, -1, width * height)
        _val, idx = torch.topk(energy, height * width // self.topk, dim=2,
            largest=True, sorted=False)
        at_sparse = torch.zeros_like(energy)
        attention_mask = at_sparse.scatter_(2, idx, 1.0)
        attention = self.softmax([energy, attention_mask])
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'key_dim': 4, 'out_dim': 4}]
