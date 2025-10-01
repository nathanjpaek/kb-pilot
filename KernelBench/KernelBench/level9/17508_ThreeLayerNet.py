import torch


class ThreeLayerNet(torch.nn.Module):

    def __init__(self, D_in, H_1, H_2, D_out):
        super(ThreeLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H_1)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(H_1, H_2)
        self.linear3 = torch.nn.Linear(H_2, D_out)

    def forward(self, data):
        hidden_1 = self.relu(self.linear1(data.float()))
        hidden_2 = self.relu(self.linear2(hidden_1))
        preds = self.linear3(hidden_2)
        return preds


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'D_in': 4, 'H_1': 4, 'H_2': 4, 'D_out': 4}]
