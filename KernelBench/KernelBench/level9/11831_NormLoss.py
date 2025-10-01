import torch


class NormLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y, Y, w, mask):
        ny = torch.linalg.norm(y, dim=1, keepdim=False) / 5.0
        nY = torch.linalg.norm(Y, dim=1, keepdim=False) / 5.0
        diff = ny - nY
        return torch.mean(torch.square(diff[mask]) * w[mask])


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.ones(
        [4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {}]
