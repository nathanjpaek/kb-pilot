import torch


class diag_offdiag_maxpool(torch.nn.Module):
    """diag_offdiag_maxpool"""

    def __init__(self):
        super(diag_offdiag_maxpool, self).__init__()

    def forward(self, inputs):
        max_diag = torch.max(torch.diagonal(inputs, dim1=-2, dim2=-1), dim=2)[0
            ]
        max_val = torch.max(max_diag)
        min_val = torch.max(torch.mul(inputs, -1))
        val = torch.abs(max_val + min_val)
        min_mat = torch.unsqueeze(torch.unsqueeze(torch.diagonal(torch.add(
            torch.mul(torch.diag_embed(inputs[0][0]), 0), val)), dim=0), dim=0)
        max_offdiag = torch.max(torch.max(torch.sub(inputs, min_mat), dim=2
            )[0], dim=2)[0]
        return torch.cat((max_diag, max_offdiag), dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
