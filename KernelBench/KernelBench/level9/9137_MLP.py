import torch


class MLP(torch.nn.Module):

    def __init__(self, input_size, ouput_size=1) ->None:
        super(MLP, self).__init__()
        self.layer_1 = torch.nn.Linear(input_size, 2 * input_size)
        self.layer_2 = torch.nn.Linear(2 * input_size, 2 * input_size)
        self.layer_3 = torch.nn.Linear(2 * input_size, input_size)
        self.layer_4 = torch.nn.Linear(input_size, int(input_size / 4))
        self.layer_out = torch.nn.Linear(int(input_size / 4), ouput_size)
        self.dropout = torch.nn.Dropout(0.3)
        self.relu = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.relu(self.layer_3(x))
        x = self.dropout(x)
        x = self.relu(self.layer_4(x))
        x = self.dropout(x)
        x = self.layer_out(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
