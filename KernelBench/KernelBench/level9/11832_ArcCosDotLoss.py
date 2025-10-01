import torch


class ArcCosDotLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y, w, mask):
        eps = 1e-12
        denom = torch.multiply(torch.linalg.norm(x, dim=1), torch.linalg.
            norm(y, dim=1)) + eps
        dot = x[:, 0, :, :] * y[:, 0, :, :] + x[:, 1, :, :] * y[:, 1, :, :]
        phasediff = torch.acos(torch.clip(dot / denom, -0.999999, 0.999999)
            ) / 3.141549
        return torch.mean(torch.square(phasediff[mask]) * w[mask])


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.ones(
        [4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {}]
