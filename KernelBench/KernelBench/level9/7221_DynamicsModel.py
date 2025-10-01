import torch


class DynamicsModel(torch.nn.Module):

    def __init__(self, D_in, D_out, hidden_unit_num):
        None
        super(DynamicsModel, self).__init__()
        self.l1 = torch.nn.Linear(D_in, hidden_unit_num)
        self.l2 = torch.nn.Linear(hidden_unit_num, D_out)
        self.logvar = torch.nn.Parameter(torch.zeros(D_out), requires_grad=True
            )

    def forward(self, X):
        mu = self.l2(torch.tanh(self.l1(X)))
        return self.l2(torch.tanh(self.l1(X))), self.logvar * torch.ones_like(
            mu)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'D_in': 4, 'D_out': 4, 'hidden_unit_num': 4}]
