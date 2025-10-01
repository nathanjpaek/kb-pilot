from _paritybench_helpers import _mock_config
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init


class Attention(nn.Module):

    def __init__(self, args, enc_dim, dec_dim, attn_dim=None):
        super(Attention, self).__init__()
        self.args = args
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.attn_dim = self.dec_dim if attn_dim is None else attn_dim
        if self.args.birnn:
            self.birnn = 2
        else:
            self.birnn = 1
        self.encoder_in = nn.Linear(self.enc_dim, self.attn_dim, bias=True)
        self.decoder_in = nn.Linear(self.dec_dim, self.attn_dim, bias=False)
        self.attn_linear = nn.Linear(self.attn_dim, 1, bias=False)
        self.init_weights()

    def init_weights(self):
        self.encoder_in.weight.data.uniform_(-0.08, 0.08)
        self.encoder_in.bias.data.fill_(0)
        self.decoder_in.weight.data.uniform_(-0.08, 0.08)
        self.attn_linear.weight.data.uniform_(-0.08, 0.08)

    def forward(self, dec_state, enc_states, mask, dag=None):
        """
        :param dec_state: 
            decoder hidden state of size batch_size x dec_dim
        :param enc_states:
            all encoder hidden states of size batch_size x max_enc_steps x enc_dim
        :param flengths:
            encoder video frame lengths of size batch_size
        """
        dec_contrib = self.decoder_in(dec_state)
        batch_size, max_enc_steps, _ = enc_states.size()
        enc_contrib = self.encoder_in(enc_states.contiguous().view(-1, self
            .enc_dim)).contiguous().view(batch_size, max_enc_steps, self.
            attn_dim)
        pre_attn = F.tanh(enc_contrib + dec_contrib.unsqueeze(1).expand_as(
            enc_contrib))
        energy = self.attn_linear(pre_attn.view(-1, self.attn_dim)).view(
            batch_size, max_enc_steps)
        alpha = F.softmax(energy, 1)
        alpha = alpha * mask
        alpha = torch.div(alpha, alpha.sum(1).unsqueeze(1).expand_as(alpha))
        context_vector = torch.bmm(alpha.unsqueeze(1), enc_states).squeeze(1)
        return context_vector, alpha


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(birnn=4), 'enc_dim': 4, 'dec_dim': 4}]
