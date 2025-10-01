from torch.nn import Module
import torch
from torch.nn import Linear
from torch.nn.init import xavier_normal_
from torch.nn.functional import relu


class ps_FNNDenoiser(Module):

    def __init__(self, input_dim):
        """The FNN enc and FNN dec of the Denoiser.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        """
        super(ps_FNNDenoiser, self).__init__()
        self._input_dim = input_dim
        self.fnn_enc = Linear(self._input_dim, int(self._input_dim / 2))
        self.fnn_dec = Linear(int(self._input_dim / 2), self._input_dim)
        self.initialize_module()

    def initialize_module(self):
        """Manual weight/bias initialization.
        """
        xavier_normal_(self.fnn_enc.weight)
        self.fnn_enc.bias.data.zero_()
        xavier_normal_(self.fnn_dec.weight)
        self.fnn_dec.bias.data.zero_()

    def forward(self, v_j_filt_prime):
        """The forward pass.

        :param v_j_filt_prime: The output of the Masker.
        :type v_j_filt_prime: torch.autograd.variable.Variable
        :return: The output of the Denoiser
        :rtype: torch.autograd.variable.Variable
        """
        fnn_enc_output = relu(self.fnn_enc(v_j_filt_prime))
        fnn_dec_output = relu(self.fnn_dec(fnn_enc_output))
        v_j_filt = fnn_dec_output.mul(v_j_filt_prime)
        return v_j_filt


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
