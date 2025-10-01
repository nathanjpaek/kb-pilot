import torch
from typing import Optional
import torch.nn as nn
from torch.nn.parameter import Parameter


class BiAttention(nn.Module):

    def __init__(self, input_size_encoder: 'int', input_size_decoder: 'int',
        num_labels: 'int', biaffine: 'bool'=True, **kwargs) ->None:
        super(BiAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.num_labels = num_labels
        self.biaffine = biaffine
        self.W_e = Parameter(torch.Tensor(self.num_labels, self.
            input_size_encoder))
        self.W_d = Parameter(torch.Tensor(self.num_labels, self.
            input_size_decoder))
        self.b = Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = Parameter(torch.Tensor(self.num_labels, self.
                input_size_decoder, self.input_size_encoder))
        else:
            self.register_parameter('U', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        nn.init.xavier_uniform_(self.W_e)
        nn.init.xavier_uniform_(self.W_d)
        nn.init.constant_(self.b, 0.0)
        if self.biaffine:
            nn.init.xavier_uniform_(self.U)

    def forward(self, input_d: 'torch.Tensor', input_e: 'torch.Tensor',
        mask_d: 'Optional[torch.Tensor]'=None, mask_e:
        'Optional[torch.Tensor]'=None) ->torch.Tensor:
        assert input_d.size(0) == input_e.size(0)
        _batch, _length_decoder, _ = input_d.size()
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
