import math
import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Parameter
import torch.utils.data


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a ** 2) * fan))
        tensor.data.uniform_(-bound, bound)


class DenseSAGEConv(torch.nn.Module):
    """See :class:`torch_geometric.nn.conv.SAGEConv`.
    """

    def __init__(self, in_channels, out_channels, normalize=False, bias=True):
        super(DenseSAGEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.lin_rel = Linear(in_channels, out_channels, bias=False)
        self.lin_root = Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x, adj, mask=None):
        """
        Args:
            x (Tensor): Node feature tensor :math:`\\mathbf{X} \\in \\mathbb{R}^{B
                \\times N \\times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\\mathbf{A} \\in \\mathbb{R}^{B
                \\times N \\times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\\mathbf{M} \\in {\\{ 0, 1 \\}}^{B \\times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()
        out = torch.matmul(adj, x)
        out = out / adj.sum(dim=-1, keepdim=True).clamp(min=1)
        out = self.lin_rel(out) + self.lin_root(x)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        if mask is not None:
            out = out * mask.view(B, N, 1)
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.
            in_channels, self.out_channels)


class Linear(torch.nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super(Linear, self).__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.weight = Parameter(Tensor(groups, in_channels // groups, 
            out_channels // groups))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform(self.weight, fan=self.weight.size(1), a=math.sqrt(5))
        uniform(self.weight.size(1), self.bias)

    def forward(self, src):
        if self.groups > 1:
            size = list(src.size())[:-1]
            src = src.view(-1, self.groups, self.in_channels // self.groups)
            src = src.transpose(0, 1).contiguous()
            out = torch.matmul(src, self.weight)
            out = out.transpose(1, 0).contiguous()
            out = out.view(*(size + [self.out_channels]))
        else:
            out = torch.matmul(src, self.weight.squeeze(0))
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, groups={}, bias={})'.format(self.__class__.
            __name__, self.in_channels, self.out_channels, self.groups, 
            self.bias is not None)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
