import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init


class Hidden2DiscreteDeal(nn.Module):

    def __init__(self, input_size, z_size, is_lstm=False, has_bias=True):
        super(Hidden2DiscreteDeal, self).__init__()
        self.z_size = z_size
        latent_size = self.z_size
        if is_lstm:
            self.p_h = nn.Linear(input_size, latent_size, bias=has_bias)
            self.p_c = nn.Linear(input_size, latent_size, bias=has_bias)
        else:
            self.p_h = nn.Linear(input_size, latent_size, bias=has_bias)
        self.is_lstm = is_lstm

    def forward(self, inputs, mask=None):
        """
        :param inputs: batch_size x input_size
        :return:
        """
        if self.is_lstm:
            h, c = inputs
            if h.dim() == 3:
                h = h.squeeze(0)
                c = c.squeeze(0)
            logits = self.p_h(h) + self.p_c(c)
        else:
            logits = self.p_h(inputs)
        log_pz = F.log_softmax(logits, dim=-1)
        return logits, log_pz


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'z_size': 4}]
