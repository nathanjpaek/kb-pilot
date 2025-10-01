import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init


class Hidden2Discrete(nn.Module):

    def __init__(self, input_size, y_size, k_size, is_lstm=False, has_bias=True
        ):
        super(Hidden2Discrete, self).__init__()
        self.y_size = y_size
        self.k_size = k_size
        latent_size = self.k_size * self.y_size
        if is_lstm:
            self.p_h = nn.Linear(input_size, latent_size, bias=has_bias)
            self.p_c = nn.Linear(input_size, latent_size, bias=has_bias)
        else:
            self.p_h = nn.Linear(input_size, latent_size, bias=has_bias)
        self.is_lstm = is_lstm

    def forward(self, inputs):
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
        logits = logits.view(-1, self.k_size)
        log_qy = F.log_softmax(logits, dim=1)
        return logits, log_qy


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'y_size': 4, 'k_size': 4}]
