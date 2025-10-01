import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.data


class Bi_Attention(nn.Module):

    def __init__(self):
        super(Bi_Attention, self).__init__()
        self.inf = 10000000000000.0

    def forward(self, in_x1, in_x2, x1_len, x2_len):
        assert in_x1.size()[0] == in_x2.size()[0]
        assert in_x1.size()[2] == in_x2.size()[2]
        assert in_x1.size()[1] == x1_len.size()[1] and in_x2.size()[1
            ] == x2_len.size()[1]
        assert in_x1.size()[0] == x1_len.size()[0] and x1_len.size()[0
            ] == x2_len.size()[0]
        batch_size = in_x1.size()[0]
        x1_max_len = in_x1.size()[1]
        x2_max_len = in_x2.size()[1]
        in_x2_t = torch.transpose(in_x2, 1, 2)
        attention_matrix = torch.bmm(in_x1, in_x2_t)
        a_mask = x1_len.le(0.5).float() * -self.inf
        a_mask = a_mask.view(batch_size, x1_max_len, -1)
        a_mask = a_mask.expand(-1, -1, x2_max_len)
        b_mask = x2_len.le(0.5).float() * -self.inf
        b_mask = b_mask.view(batch_size, -1, x2_max_len)
        b_mask = b_mask.expand(-1, x1_max_len, -1)
        attention_a = F.softmax(attention_matrix + a_mask, dim=2)
        attention_b = F.softmax(attention_matrix + b_mask, dim=1)
        out_x1 = torch.bmm(attention_a, in_x2)
        attention_b_t = torch.transpose(attention_b, 1, 2)
        out_x2 = torch.bmm(attention_b_t, in_x1)
        return out_x1, out_x2


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4,
        4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
