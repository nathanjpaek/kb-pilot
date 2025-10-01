import torch
import torch.nn as nn


class Combiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T}, m{t:T}, s)`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T} and m_{t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)
    """

    def __init__(self, z_dim, static_dim, rnn_dim):
        super().__init__()
        self.concat_dim = z_dim + static_dim
        self.lin_z_to_hidden = nn.Linear(self.concat_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, mini_batch_static, h_rnn):
        """
        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{t:T}, m{t:T}, s)`
        """
        concat = torch.cat((z_t_1, mini_batch_static), dim=1)
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(concat)) + h_rnn)
        loc = self.lin_hidden_to_loc(h_combined)
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        return loc, scale


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'z_dim': 4, 'static_dim': 4, 'rnn_dim': 4}]
