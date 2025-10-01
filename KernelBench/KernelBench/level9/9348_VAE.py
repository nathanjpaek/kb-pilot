import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, x_dim, h_dim1, h_dim2, h_dim3, z_dim):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, h_dim3)
        self.fc31 = nn.Linear(h_dim3, z_dim)
        self.fc32 = nn.Linear(h_dim3, z_dim)
        self.dropout = nn.Dropout(0.5)
        self.Encoder = nn.Sequential(self.fc1, nn.Dropout(0.2), nn.ReLU(),
            self.fc2, nn.Dropout(0.2), nn.ReLU(), self.fc3, nn.Dropout(0.2),
            nn.ReLU())
        self.fc4 = nn.Linear(z_dim, h_dim3)
        self.fc5 = nn.Linear(h_dim3, h_dim2)
        self.fc6 = nn.Linear(h_dim2, h_dim1)
        self.fc7 = nn.Linear(h_dim1, x_dim)
        self.Decoder = nn.Sequential(self.fc4, nn.Dropout(0.2), nn.
            LeakyReLU(negative_slope=0.2), self.fc5, nn.Dropout(0.2), nn.
            LeakyReLU(negative_slope=0.2), self.fc6, nn.Dropout(0.2), nn.
            LeakyReLU(negative_slope=0.2), self.fc7, nn.Dropout(0.2), nn.
            LeakyReLU(negative_slope=0.2))

    def encode(self, x):
        h = self.Encoder(x)
        return self.fc31(h), self.fc32(h)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        output = self.Decoder(z)
        return output

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, self.x_dim))
        z = self.sampling(mu, log_var)
        return self.decode(z), mu, log_var


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'x_dim': 4, 'h_dim1': 4, 'h_dim2': 4, 'h_dim3': 4, 'z_dim': 4}
        ]
