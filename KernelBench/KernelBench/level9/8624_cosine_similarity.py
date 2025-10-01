from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class cosine_similarity(nn.Module):

    def __init__(self, args):
        super(cosine_similarity, self).__init__()
        self.row_wise_avgpool = nn.AvgPool1d(kernel_size=3, stride=1)

    def forward(self, x, y):
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        cos_mat = torch.softmax(torch.bmm(y, x.transpose(1, 2)), dim=2)
        cos_mat = self.row_wise_avgpool(cos_mat)
        cos_vec, _t = torch.max(cos_mat, dim=2)
        cos_vec = torch.mean(cos_vec, dim=1)
        None
        return cos_vec


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config()}]
