import torch
import torch.nn as nn
import torch.utils.data.dataloader
from torch.nn import Parameter
from torch.nn.parameter import Parameter
import torch.nn


class BiaffineAttention(nn.Module):
    """
    Adopted from NeuroNLP2:
        https://github.com/XuezheMax/NeuroNLP2/blob/master/neuronlp2/nn/modules/attention.py

    Bi-Affine attention layer.
    """

    def __init__(self, input_size_encoder, input_size_decoder, hidden_size=
        150, num_labels=1, biaffine=True, **kwargs):
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
        super(BiaffineAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.hidden_size = hidden_size
        self.linear_encoder = torch.nn.Linear(self.input_size_encoder, self
            .hidden_size)
        self.linear_decoder = torch.nn.Linear(self.input_size_decoder, self
            .hidden_size)
        self.num_labels = num_labels
        self.biaffine = biaffine
        self.W_d = Parameter(torch.Tensor(self.num_labels, self.hidden_size))
        self.W_e = Parameter(torch.Tensor(self.num_labels, self.hidden_size))
        self.b = Parameter(torch.Tensor(1, self.num_labels))
        if self.biaffine:
            self.U = Parameter(torch.Tensor(self.num_labels, self.
                hidden_size, self.hidden_size))
        else:
            self.register_parameter('U', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W_d)
        nn.init.xavier_normal_(self.W_e)
        nn.init.constant_(self.b, 0.0)
        if self.biaffine:
            nn.init.xavier_normal_(self.U)

    def forward(self, input_s, input_t, mask_d=None, mask_e=None):
        """
        Args:
            input_s: Tensor
                the student input tensor with shape = [batch, input_size]
            input_t: Tensor
                the teacher input tensor with shape = [batch, num_teachers, input_size]
            mask_d: None
            mask_e: None
        Returns: Tensor
            the energy tensor with shape = [batch, length]
        """
        assert input_s.size(0) == input_t.size(0
            ), 'batch sizes of encoder and decoder are requires to be equal.'
        input_s.size()
        _, _num_teachers, _ = input_t.size()
        input_s = self.linear_encoder(input_s)
        input_t = self.linear_decoder(input_t)
        out_e = torch.matmul(self.W_e, input_s.transpose(1, 0)).transpose(1, 0)
        out_d = torch.einsum('nd,bnd->bn', self.W_d, input_t)
        if self.biaffine:
            output = torch.einsum('bd,nde->bne', input_s, self.U)
            output = torch.einsum('bne,bne->bn', output, input_t)
            output = output + out_d + out_e + self.b
        else:
            output = out_d + out_d + self.b
        if mask_d is not None and mask_e is not None:
            output = output * mask_d.unsqueeze(1).unsqueeze(3
                ) * mask_e.unsqueeze(1).unsqueeze(2)
        output = torch.nn.functional.softmax(output, 1)
        return output


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size_encoder': 4, 'input_size_decoder': 4}]
