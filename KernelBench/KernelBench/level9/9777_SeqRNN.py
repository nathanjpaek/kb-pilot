import torch
import torch.nn as nn


class SeqRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(SeqRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(in_features=input_size + hidden_size,
            out_features=hidden_size)
        self.i2o = nn.Linear(in_features=input_size + hidden_size,
            out_features=output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim=1)
        hidden = self.i2h(combined)
        out = self.i2o(combined)
        out = self.softmax(out)
        return out, hidden

    def init_hidden(self, batch):
        return torch.zeros(batch, self.hidden_size, device=device)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'output_size': 4,
        'n_layers': 1}]
