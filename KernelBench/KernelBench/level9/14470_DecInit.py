import torch
import torch.nn as nn


class DecInit(nn.Module):

    def __init__(self, d_enc, d_dec, n_enc_layer):
        self.d_enc_model = d_enc
        self.n_enc_layer = n_enc_layer
        self.d_dec_model = d_dec
        super(DecInit, self).__init__()
        self.initer = nn.Linear(self.d_enc_model * self.n_enc_layer, self.
            d_dec_model)
        self.tanh = nn.Tanh()

    def forward(self, hidden):
        if isinstance(hidden, tuple) or isinstance(hidden, list) or hidden.dim(
            ) == 3:
            hidden = [h for h in hidden]
            hidden = torch.cat(hidden, dim=1)
        hidden = hidden.contiguous().view(hidden.size(0), -1)
        return self.tanh(self.initer(hidden))


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_enc': 4, 'd_dec': 4, 'n_enc_layer': 1}]
