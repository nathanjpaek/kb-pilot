import torch
import torch.nn as nn


class BiaffineAttention(nn.Module):

    def __init__(self, in1_features, in2_features, num_label, bias=True):
        super(BiaffineAttention, self).__init__()
        self.bilinear = nn.Bilinear(in1_features, in2_features, num_label,
            bias=bias)
        self.linear = nn.Linear(in1_features + in2_features, num_label,
            bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, head, dep):
        """
        :param head: [batch, seq_len, hidden] 输入特征1, 即label-head
        :param dep: [batch, seq_len, hidden] 输入特征2, 即label-dep
        :return output: [batch, seq_len, num_cls] 每个元素对应类别的概率图
        """
        output = self.bilinear(head, dep)
        biaff_score = output + self.linear(torch.cat((head, dep), dim=-1))
        biaff_score = biaff_score.transpose(1, 2)
        att_weigths = self.softmax(biaff_score)
        att_out = torch.bmm(att_weigths, dep)
        return att_out


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in1_features': 4, 'in2_features': 4, 'num_label': 4}]
