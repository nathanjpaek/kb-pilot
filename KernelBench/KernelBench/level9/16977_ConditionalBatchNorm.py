import torch


class ConditionalBatchNorm(torch.nn.Module):

    def __init__(self, no, z_dim):
        super().__init__()
        self.no = no
        self.bn = torch.nn.InstanceNorm2d(no, affine=False)
        self.condition = torch.nn.Linear(z_dim, 2 * no)

    def forward(self, x, z):
        cond = self.condition(z).view(-1, 2 * self.no, 1, 1)
        return self.bn(x) * cond[:, :self.no] + cond[:, self.no:]


def get_inputs():
    return [torch.rand([64, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'no': 4, 'z_dim': 4}]
