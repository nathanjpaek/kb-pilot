import torch


class WorldNet(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WorldNet, self).__init__()
        self.fc_in = torch.nn.Linear(input_dim, hidden_dim)
        self.fc_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, output_dim * 2)
        den = 2 * torch.tensor(input_dim).float().sqrt()
        torch.nn.init.normal_(self.fc_in.weight, std=1.0 / den)
        torch.nn.init.normal_(self.fc_1.weight, std=1.0 / den)
        torch.nn.init.normal_(self.fc_2.weight, std=1.0 / den)
        torch.nn.init.normal_(self.fc_3.weight, std=1.0 / den)
        torch.nn.init.normal_(self.fc_out.weight, std=1.0 / den)

    def forward(self, x):
        out = self.swish(self.fc_in(x))
        out = self.swish(self.fc_1(out))
        out = self.swish(self.fc_2(out))
        out = self.swish(self.fc_3(out))
        out = self.fc_out(out)
        return out

    def swish(self, x):
        return x * torch.sigmoid(x)

    def get_decays(self):
        decays = 2.5e-05 * (self.fc_in.weight ** 2).sum() / 2.0
        decays += 5e-05 * (self.fc_1.weight ** 2).sum() / 2.0
        decays += 7.5e-05 * (self.fc_2.weight ** 2).sum() / 2.0
        decays += 7.5e-05 * (self.fc_3.weight ** 2).sum() / 2.0
        decays += 0.0001 * (self.fc_out.weight ** 2).sum() / 2.0
        return decays


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4}]
