import torch
import torch.nn as nn


class DecoderBias(nn.Module):

    def __init__(self, dim1_batch, latent_dim, bias=False):
        super().__init__()
        self.dim1_latent_decoder = nn.Parameter(torch.randn(latent_dim,
            latent_dim))
        self.dim2_latent_decoder = nn.Parameter(torch.randn(latent_dim,
            latent_dim))
        self.batch_decoder_weight = nn.Parameter(torch.randn(dim1_batch,
            latent_dim))
        if bias:
            self.dim1_bias = nn.Parameter(torch.randn(latent_dim))
            self.dim2_bias = nn.Parameter(torch.randn(latent_dim))
        else:
            self.dim1_bias = 0
            self.dim2_bias = 0

    def forward(self, x):
        self.dim1_decoder_weight = torch.cat([self.dim1_latent_decoder,
            self.batch_decoder_weight], dim=0)
        self.dim2_decoder_weight = torch.cat([self.dim2_latent_decoder,
            self.batch_decoder_weight], dim=0)
        dim1_latent = x[0]
        dim2_latent = x[1]
        batch = x[2]
        dim1_input = torch.cat([dim1_latent, batch], dim=1)
        dim2_input = torch.cat([dim2_latent, batch], dim=1)
        dim1_output = dim1_input @ self.dim1_decoder_weight + self.dim1_bias
        dim2_output = dim2_input @ self.dim2_decoder_weight + self.dim2_bias
        return dim1_output, dim2_output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim1_batch': 4, 'latent_dim': 4}]
