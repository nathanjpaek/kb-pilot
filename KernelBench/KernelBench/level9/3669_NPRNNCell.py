import torch
from torch import nn


class NPRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, clip=2.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.clip = clip
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.alpha = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.norm = nn.LayerNorm(hidden_size)
        self.modulator = nn.Linear(hidden_size, 1)
        self.modfanout = nn.Linear(1, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.001)
        nn.init.normal_(self.alpha, std=0.001)

    def forward(self, x, h_pre, hebb):
        weight = self.weight + self.alpha * hebb
        h_post = self.fc_in(x) + (h_pre.unsqueeze(1) @ weight).squeeze(1)
        h_post = torch.tanh(self.norm(h_post))
        out = self.fc_out(h_post)
        m = torch.tanh(self.modulator(h_post))
        eta = self.modfanout(m.unsqueeze(2))
        delta = eta * (h_pre.unsqueeze(2) @ h_post.unsqueeze(1))
        hebb = torch.clamp(hebb + delta, min=-self.clip, max=self.clip)
        return out, h_post, m, hebb


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}]
