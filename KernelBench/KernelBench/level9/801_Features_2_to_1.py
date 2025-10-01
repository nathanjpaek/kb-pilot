import torch
import torch.optim
import torch.nn as nn


class Features_2_to_1(nn.Module):

    def __init__(self):
        """
        take a batch (bs, n_vertices, n_vertices, in_features)
        and returns (bs, n_vertices, basis * in_features)
        where basis = 5
        """
        super().__init__()

    def forward(self, x):
        b, n, _, in_features = x.size()
        basis = 5
        diag_part = torch.diagonal(x, dim1=1, dim2=2).permute(0, 2, 1)
        max_diag_part = torch.max(diag_part, 1)[0].unsqueeze(1)
        max_of_rows = torch.max(x, 2)[0]
        max_of_cols = torch.max(x, 1)[0]
        max_all = torch.max(torch.max(x, 1)[0], 1)[0].unsqueeze(1)
        op1 = diag_part
        op2 = max_diag_part.expand_as(op1)
        op3 = max_of_rows
        op4 = max_of_cols
        op5 = max_all.expand_as(op1)
        output = torch.stack([op1, op2, op3, op4, op5], dim=2)
        assert output.size() == (b, n, basis, in_features), output.size()
        return output.view(b, n, basis * in_features)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
