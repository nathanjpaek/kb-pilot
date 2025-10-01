import torch


class Backbone(torch.nn.Module):

    def __init__(self, input_size=4, hidden_size=10, latent_size=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.dense1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.dense2 = torch.nn.Linear(self.hidden_size, self.latent_size)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.dense1(x))
        out = self.activation(self.dense2(x))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
