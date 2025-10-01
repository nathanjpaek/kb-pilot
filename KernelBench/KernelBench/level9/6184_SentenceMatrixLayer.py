import torch
import torch.nn as nn


class SentenceMatrixLayer(nn.Module):

    def __init__(self, in_size, out_size=1, p_Asem=0.6):
        super(SentenceMatrixLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.p_Asem = p_Asem
        self.linear = nn.Linear(in_size * 2, out_size)

    def forward(self, x, adj, mask):
        seq_len = x.shape[1]
        xi = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        xj = x.unsqueeze(1).expand(-1, seq_len, -1, -1)
        xij = torch.sigmoid(self.linear(torch.cat((xi, xj), dim=-1))).squeeze(
            -1)
        A_esm = self.p_Asem * xij + (1 - self.p_Asem) * adj
        assert mask.shape[1] == seq_len, 'seq_len inconsistent'
        mask_i = mask.unsqueeze(1).expand(-1, seq_len, -1)
        mask_j = mask.unsqueeze(2).expand(-1, -1, seq_len)
        A_mask = mask_i * mask_j
        return A_esm.masked_fill(A_mask == 0, 1e-09)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4])
        ]


def get_init_inputs():
    return [[], {'in_size': 4}]
