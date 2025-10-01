import torch


class SoftHistogram(torch.nn.Module):
    """
    Motivated by https://discuss.pytorch.org/t/differentiable-torch-histc/25865/3
    """

    def __init__(self, bins, min_bin_edge, max_bin_edge, sigma):
        super(SoftHistogram, self).__init__()
        self.sigma = sigma
        self.delta = float(max_bin_edge - min_bin_edge) / float(bins)
        self.centers = float(min_bin_edge) + self.delta * (torch.arange(
            bins).float() + 0.5)
        self.centers = torch.nn.Parameter(self.centers, requires_grad=False)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(
            self.sigma * (x - self.delta / 2))
        x = x.sum(dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'bins': 4, 'min_bin_edge': 4, 'max_bin_edge': 4, 'sigma': 4}]
