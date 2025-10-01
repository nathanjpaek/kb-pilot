import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter
import torch.utils.data


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


class DenseSAGEConv(torch.nn.Module):
    """See :class:`torch_geometric.nn.conv.SAGEConv`.

    :rtype: :class:`Tensor`
    """

    def __init__(self, in_channels, out_channels, normalize=False, bias=True):
        super(DenseSAGEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)

    def forward(self, x, adj, mask=None, add_loop=True):
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
            mask (ByteTensor, optional): Mask matrix
                :math:`\\mathbf{M} \\in {\\{ 0, 1 \\}}^{B \\times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1
        out = torch.matmul(adj, x)
        out = out / adj.sum(dim=-1, keepdim=True).clamp(min=1)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            out = out + self.bias
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        if mask is not None:
            out = out * mask.view(B, N, 1)
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.
            in_channels, self.out_channels)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
