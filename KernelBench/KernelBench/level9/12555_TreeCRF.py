import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter


def logdet(x):
    """

    Args:
        x: 2D positive semidefinite matrix.

    Returns: log determinant of x

    """
    None
    None
    u_chol = x.potrf()
    return torch.sum(torch.log(u_chol.diag())) * 2


class BiAAttention(nn.Module):
    """
    Bi-Affine attention layer.
    """

    def __init__(self, input_size_encoder, input_size_decoder, num_labels,
        biaffine=True, **kwargs):
        """

        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        """
        super(BiAAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.num_labels = num_labels
        self.biaffine = biaffine
        self.W_d = Parameter(torch.Tensor(self.num_labels, self.
            input_size_decoder))
        self.W_e = Parameter(torch.Tensor(self.num_labels, self.
            input_size_encoder))
        self.b = Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = Parameter(torch.Tensor(self.num_labels, self.
                input_size_decoder, self.input_size_encoder))
        else:
            self.register_parameter('U', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_d)
        nn.init.xavier_uniform_(self.W_e)
        nn.init.constant_(self.b, 0.0)
        if self.biaffine:
            nn.init.xavier_uniform_(self.U)

    def forward(self, input_d, input_e, mask_d=None, mask_e=None):
        """

        Args:
            input_d: Tensor
                the decoder input tensor with shape = [batch_size, length_decoder, input_size]
            input_e: Tensor
                the child input tensor with shape = [batch_size, length_encoder, input_size]
            mask_d: Tensor or None
                the mask tensor for decoder with shape = [batch_size, length_decoder]
            mask_e: Tensor or None
                the mask tensor for encoder with shape = [batch_size, length_encoder]

        Returns: Tensor
            the energy tensor with shape = [batch_size, num_label, length, length]

        """
        assert input_d.size(0) == input_e.size(0
            ), 'batch sizes of encoder and decoder are requires to be equal.'
        _batch_size, _length_decoder, _ = input_d.size()
        _, _length_encoder, _ = input_e.size()
        out_d = torch.matmul(self.W_d, input_d.transpose(1, 2)).unsqueeze(3)
        out_e = torch.matmul(self.W_e, input_e.transpose(1, 2)).unsqueeze(2)
        if self.biaffine:
            output = torch.matmul(input_d.unsqueeze(1), self.U)
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))
            output = output + out_d + out_e + self.b
        else:
            output = out_d + out_d + self.b
        if mask_d is not None:
            output = output * mask_d.unsqueeze(1).unsqueeze(3
                ) * mask_e.unsqueeze(1).unsqueeze(2)
        return output


class TreeCRF(nn.Module):
    """
    Tree CRF layer.
    """

    def __init__(self, input_size, num_labels, biaffine=True):
        """

        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        """
        super(TreeCRF, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels
        self.attention = BiAAttention(input_size, input_size, num_labels,
            biaffine=biaffine)

    def forward(self, input_h, input_c, mask=None):
        """

        Args:
            input_h: Tensor
                the head input tensor with shape = [batch_size, length, input_size]
            input_c: Tensor
                the child input tensor with shape = [batch_size, length, input_size]
            mask: Tensor or None
                the mask tensor with shape = [batch_size, length]
            lengths: Tensor or None
                the length tensor with shape = [batch_size]

        Returns: Tensor
            the energy tensor with shape = [batch_size, num_label, length, length]

        """
        _, length, _ = input_h.size()
        output = self.attention(input_h, input_c, mask_d=mask, mask_e=mask)
        output = output + torch.diag(output.data.new(length).fill_(-np.inf))
        return output

    def loss(self, input_h, input_c, heads, arc_tags, mask=None, lengths=None):
        """

        Args:
            input_h: Tensor
                the head input tensor with shape = [batch_size, length, input_size]
            input_c: Tensor
                the child input tensor with shape = [batch_size, length, input_size]
            target: Tensor
                the tensor of target labels with shape [batch_size, length]
            mask:Tensor or None
                the mask tensor with shape = [batch_size, length]
            lengths: tensor or list of int
                the length of each input shape = [batch_size]

        Returns: Tensor
                A 1D tensor for minus log likelihood loss
        """
        batch_size, length, _ = input_h.size()
        energy = self.forward(input_h, input_c, mask=mask)
        A = torch.exp(energy)
        if mask is not None:
            A = A * mask.unsqueeze(1).unsqueeze(3) * mask.unsqueeze(1
                ).unsqueeze(2)
        A = A.sum(dim=1)
        D = A.sum(dim=1, keepdim=True)
        rtol = 0.0001
        atol = 1e-06
        D += D * rtol + atol
        D = A.data.new(A.size()).zero_() + D
        D = D * torch.eye(length).type_as(D)
        L = D - A
        if lengths is None:
            if mask is None:
                lengths = [length for _ in range(batch_size)]
            else:
                lengths = mask.data.sum(dim=1).long()
        z = energy.data.new(batch_size)
        for b in range(batch_size):
            Lx = L[b, 1:lengths[b], 1:lengths[b]]
            z[b] = logdet(Lx)
        index = torch.arange(0, length).view(length, 1).expand(length,
            batch_size)
        index = index.type_as(energy.data).long()
        batch_index = torch.arange(0, batch_size).type_as(energy.data).long()
        tgt_energy = energy[batch_index, arc_tags.data.t(), heads.data.t(),
            index][1:]
        tgt_energy = tgt_energy.sum(dim=0)
        return z - tgt_energy


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'num_labels': 4}]
