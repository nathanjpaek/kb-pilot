import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ConcatAttention(nn.Module):
    """
    Concatenate attention layer.
    """

    def __init__(self, input_size_encoder, input_size_decoder, hidden_size,
        num_labels, **kwargs):
        """

        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            hidden_size: int
                the dimension of the hidden.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        """
        super(ConcatAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.W_d = Parameter(torch.Tensor(self.input_size_decoder, self.
            hidden_size))
        self.W_e = Parameter(torch.Tensor(self.input_size_encoder, self.
            hidden_size))
        self.b = Parameter(torch.Tensor(self.hidden_size))
        self.v = Parameter(torch.Tensor(self.hidden_size, self.num_labels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform(self.W_d)
        nn.init.xavier_uniform(self.W_e)
        nn.init.xavier_uniform(self.v)
        nn.init.constant(self.b, 0.0)

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
        out_d = torch.matmul(input_d, self.W_d).unsqueeze(1)
        out_e = torch.matmul(input_e, self.W_e).unsqueeze(2)
        out = torch.tanh(out_d + out_e + self.b)
        return torch.matmul(out, self.v).transpose(1, 3)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size_encoder': 4, 'input_size_decoder': 4,
        'hidden_size': 4, 'num_labels': 4}]
