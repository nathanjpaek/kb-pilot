import torch


class AgentNN(torch.nn.Module):
    """ Simple network. """

    def __init__(self, D_in, D_out):
        super(AgentNN, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, 20)
        self.h1 = torch.nn.Linear(20, 15)
        self.linear2 = torch.nn.Linear(15, D_out)
        self.activation = torch.nn.Tanh()

    def forward(self, x, mask=None):
        x = self.activation(self.linear1(x.float()))
        x = self.activation(self.h1(x))
        y_pred = self.activation(self.linear2(x))
        if mask is not None:
            y_pred[mask == 0] = 0
        return y_pred

    def save_model(self, path: 'str'=None) ->None:
        if not path:
            path = 'agent_model.pt'
        torch.save(self.state_dict(), path)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'D_in': 4, 'D_out': 4}]
