import torch
import torch.nn as nn


class PreProcess(nn.Module):

    def __init__(self):
        """
        Blocco di pre-processing delle immagini. Prende il tensore in ingresso nella forma
        (batch, width, height, channel), lo permuta e lo normalizza tra 0 e 1.
        """
        super(PreProcess, self).__init__()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x.float()
        x = x.div(255.0)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
