import torch
import torch.random


class Join(torch.nn.Module):
    """Join layer
    """

    def forward(self, unary: 'torch.Tensor', binary: 'torch.Tensor', index1:
        'torch.Tensor', index2: 'torch.Tensor'):
        """Join the unary and binary tensors.
        :param unary: [u, |U|] the tensor with unary predicates pre-activations
        :param binary: [b, |B|] the tensor with binary predicates pre-activations
        :param index1: [b] a vector containing the indices of the first object
        of the pair referred by binary tensor
        :param index1: [b] a vector containing the indices of the second object
        of the pair referred by binary tensor
        :returns [b, 2|U| + |B|]
        """
        index1 = torch.squeeze(index1)
        index2 = torch.squeeze(index2)
        if index1.ndim == 0 and index2.ndim == 0:
            index1 = torch.unsqueeze(index1, 0)
            index2 = torch.unsqueeze(index2, 0)
        u1 = unary[index1]
        u2 = unary[index2]
        return torch.cat([u1, u2, binary], dim=1)


def get_inputs():
    return [torch.ones([4, 4], dtype=torch.int64), torch.rand([4, 4]),
        torch.ones([4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {}]
