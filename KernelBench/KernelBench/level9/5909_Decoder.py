import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class Decoder(nn.Module):

    def __init__(self, num_question, k_3, k_4, dropout_rate):
        super(Decoder, self).__init__()
        self.layer_2 = nn.Linear(k_4, num_question)
        self.dropout = nn.Dropout(dropout_rate)

    def get_weight_norm(self):
        """ Return ||W||

        :return: float
        """
        layer_2_w_norm = torch.norm(self.layer_2.weight, 2)
        return layer_2_w_norm

    def forward(self, inputs):
        out = inputs
        out = self.dropout(out)
        out = self.layer_2(out)
        out = F.sigmoid(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_question': 4, 'k_3': 4, 'k_4': 4, 'dropout_rate': 0.5}]
