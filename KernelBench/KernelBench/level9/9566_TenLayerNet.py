import torch


class TenLayerNet(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super(TenLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, H)
        self.linear6 = torch.nn.Linear(H, H)
        self.linear7 = torch.nn.Linear(H, H)
        self.linear8 = torch.nn.Linear(H, H)
        self.linear9 = torch.nn.Linear(H, H)
        self.linear10 = torch.nn.Linear(H, D_out)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.relu(self.linear3(x))
        x = self.dropout(x)
        x = self.relu(self.linear4(x))
        x = self.dropout(x)
        x = self.relu(self.linear5(x))
        x = self.dropout(x)
        x = self.relu(self.linear6(x))
        x = self.dropout(x)
        x = self.relu(self.linear7(x))
        x = self.dropout(x)
        x = self.relu(self.linear8(x))
        x = self.dropout(x)
        x = self.relu(self.linear9(x))
        x = self.dropout(x)
        y_pred = self.linear10(x)
        return y_pred


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'D_in': 4, 'H': 4, 'D_out': 4}]
