from _paritybench_helpers import _mock_config
import torch
import torch.utils.data
import torch.nn as nn


def batch_product(iput, mat2):
    result = None
    for i in range(iput.size()[0]):
        op = torch.mm(iput[i], mat2)
        op = op.unsqueeze(0)
        if result is None:
            result = op
        else:
            result = torch.cat((result, op), 0)
    return result.squeeze(2)


class rec_attention(nn.Module):

    def __init__(self, hm, args):
        super(rec_attention, self).__init__()
        self.num_directions = 2 if args.bidirectional else 1
        if hm is False:
            self.bin_rep_size = args.bin_rnn_size * self.num_directions
        else:
            self.bin_rep_size = args.bin_rnn_size
        self.bin_context_vector = nn.Parameter(torch.Tensor(self.
            bin_rep_size, 1), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)
        self.bin_context_vector.data.uniform_(-0.1, 0.1)

    def forward(self, iput):
        alpha = self.softmax(batch_product(iput, self.bin_context_vector))
        [batch_size, source_length, _bin_rep_size2] = iput.size()
        repres = torch.bmm(alpha.unsqueeze(2).view(batch_size, -1,
            source_length), iput)
        return repres, alpha


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hm': 4, 'args': _mock_config(bidirectional=4,
        bin_rnn_size=4)}]
