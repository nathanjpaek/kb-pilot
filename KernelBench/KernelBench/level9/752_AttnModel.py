import torch
from torch import nn
import torch.nn.functional as F


class AttnModel(nn.Module):
    """
    Attention model
    """

    def __init__(self, inp_size, out_size=None, att_type='dot'):
        """
        :param inp_size: Input size on which the the attention
        :param out_size: Output of attention
        """
        super(AttnModel, self).__init__()
        self.inp_size = inp_size
        self.out_size = out_size if out_size is not None else inp_size
        if att_type == 'dot':
            assert self.inp_size == self.out_size
        elif att_type == 'general':
            self.attn_W = nn.Linear(self.inp_size, self.out_size)
        self.attn_type = att_type
        self.attn_func = {'dot': self.dot_attn, 'general': self.general_attn}[
            self.attn_type]

    @staticmethod
    def dot_attn(this_rnn_out, encoder_outs):
        this_run_out = this_rnn_out.unsqueeze(1).expand_as(encoder_outs)
        weights = (encoder_outs * this_run_out).sum(dim=-1)
        return weights

    def general_attn(self, this_rnn_out, encoder_outs):
        mapped_enc_outs = self.attn_W(encoder_outs)
        return self.dot_attn(this_rnn_out, mapped_enc_outs)

    def forward(self, this_rnn_out, encoder_outs):
        assert encoder_outs.shape[-1] == self.inp_size
        assert this_rnn_out.shape[-1] == self.out_size
        weights = self.attn_func(this_rnn_out, encoder_outs)
        return F.softmax(weights, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inp_size': 4}]
