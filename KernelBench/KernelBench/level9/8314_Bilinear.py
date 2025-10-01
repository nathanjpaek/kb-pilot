import torch
import torch.nn as nn


class Bilinear(nn.Module):

    def __init__(self, in_dim1, in_dim2, label_dim=1, use_input_bias=False):
        super(Bilinear, self).__init__()
        self.label_dim = label_dim
        self.use_input_bias = use_input_bias
        if self.use_input_bias:
            in_dim1 += 1
            in_dim2 += 1
        self.U = nn.Parameter(torch.randn(label_dim, in_dim1, in_dim2))
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.U)
        nn.init.zeros_(self.bias)

    def forward(self, x1, x2):
        """
        :param x1: (bs, len1, in_dim1)
        :param x2: (bs, len2, in_dim2)
        :return: (bs, len1, len2, label_dim)
        """
        if self.use_input_bias:
            bias1 = x1.new_ones(x1.size()[:-1] + (1,))
            bias2 = x2.new_ones(x2.size()[:-1] + (1,))
            x1 = torch.cat((x1, bias1), dim=-1)
            x2 = torch.cat((x2, bias2), dim=-1)
        tmp = torch.matmul(x1.unsqueeze(1), self.U)
        out = torch.matmul(tmp, x2.unsqueeze(1).transpose(2, 3).contiguous())
        final = out.squeeze(1) + self.bias
        if self.label_dim > 1:
            final = final.permute(0, 2, 3, 1)
        return final


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim1': 4, 'in_dim2': 4}]
