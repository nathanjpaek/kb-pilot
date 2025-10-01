import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):

    def __init__(self, in_features, att_hidden, out_features, bias=True):
        super(AdditiveAttention, self).__init__()
        self.out_size = out_features
        self.linear1 = nn.Linear(in_features=in_features, out_features=
            att_hidden, bias=bias)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(in_features=att_hidden, out_features=
            out_features, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, mask=None):
        """
        :param inputs: (bz, seq_len, in_features)
        :param mask: (bz, seq_len)  填充为0
        :return:
        """
        add_score = self.linear2(self.tanh(self.linear1(inputs)))
        add_score = add_score.transpose(1, 2)
        if mask is not None:
            pad_mask = mask == 0
            add_score = add_score.masked_fill(pad_mask[:, None, :], -
                1000000000.0)
        att_weights = self.softmax(add_score)
        att_out = torch.bmm(att_weights, inputs)
        return att_out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'att_hidden': 4, 'out_features': 4}]
