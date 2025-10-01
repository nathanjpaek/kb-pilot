import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


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


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size_encoder': 4, 'input_size_decoder': 4,
        'num_labels': 4}]
