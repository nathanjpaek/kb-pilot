import torch
from torch import nn
import torch.utils.data
from torch.nn import Parameter


class BiAffine(nn.Module):

    def __init__(self, n_enc, n_dec, n_labels, biaffine=True, **kwargs):
        """

        :param int n_enc: the dimension of the encoder input.
        :param int n_dec: the dimension of the decoder input.
        :param int n_labels: the number of labels of the crf layer
        :param bool biaffine: if apply bi-affine parameter.
        """
        super(BiAffine, self).__init__()
        self.n_enc = n_enc
        self.n_dec = n_dec
        self.num_labels = n_labels
        self.biaffine = biaffine
        self.W_d = Parameter(torch.Tensor(self.num_labels, self.n_dec))
        self.W_e = Parameter(torch.Tensor(self.num_labels, self.n_enc))
        self.b = Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = Parameter(torch.Tensor(self.num_labels, self.n_dec,
                self.n_enc))
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

        :param Tensor input_d: the decoder input tensor with shape = [batch, length_decoder, input_size]
        :param Tensor input_e: the child input tensor with shape = [batch, length_encoder, input_size]
        :param mask_d: Tensor or None, the mask tensor for decoder with shape = [batch, length_decoder]
        :param mask_e: Tensor or None, the mask tensor for encoder with shape = [batch, length_encoder]
        :returns: Tensor, the energy tensor with shape = [batch, num_label, length, length]
        """
        assert input_d.size(0) == input_e.size(0
            ), 'batch sizes of encoder and decoder are requires to be equal.'
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
    return [[], {'n_enc': 4, 'n_dec': 4, 'n_labels': 4}]
