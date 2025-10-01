import torch


class DiffLoss(torch.nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, D1, D2):
        D1 = D1.view(D1.size(0), -1)
        D1_norm = torch.norm(D1, p=2, dim=1, keepdim=True).detach()
        D1_norm = D1.div(D1_norm.expand_as(D1) + 1e-06)
        D2 = D2.view(D2.size(0), -1)
        D2_norm = torch.norm(D2, p=2, dim=1, keepdim=True).detach()
        D2_norm = D2.div(D2_norm.expand_as(D2) + 1e-06)
        return torch.mean(D1_norm.mm(D2_norm.t()).pow(2))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
