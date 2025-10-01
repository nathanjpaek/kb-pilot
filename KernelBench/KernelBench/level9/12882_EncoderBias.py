import torch
import torch.nn as nn


class EncoderBias(nn.Module):

    def __init__(self, input_dim1, input_dim2, batch_feature, latent_dim,
        bias=False):
        """[summary]
        Args:
            input_dim1 ([type]): [mod1 dimemsion]
            input_dim2 ([type]): [mod2 dimemsion]
            batch_feature ([type]): [batch dimemsion]
            latent_dim ([type]): [latent dimemsion]
            bias (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.dim1_encoder_weight = nn.Parameter(torch.randn(input_dim1,
            latent_dim))
        self.dim2_encoder_weight = nn.Parameter(torch.randn(input_dim2,
            latent_dim))
        self.batch_encoder_weight = nn.Parameter(torch.randn(batch_feature,
            latent_dim))
        if bias:
            self.dim1_bias = nn.Parameter(torch.randn(latent_dim))
            self.dim2_bias = nn.Parameter(torch.randn(latent_dim))
        else:
            self.dim1_bias = 0
            self.dim2_bias = 0

    def forward(self, x):
        self.dim1_weight = torch.cat([self.dim1_encoder_weight, self.
            batch_encoder_weight], dim=0)
        self.dim2_weight = torch.cat([self.dim2_encoder_weight, self.
            batch_encoder_weight], dim=0)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        dim1_input = torch.cat([x1, x3], dim=1)
        dim2_input = torch.cat([x2, x3], dim=1)
        dim1_encoder = dim1_input @ self.dim1_weight + self.dim1_bias
        dim2_encoder = dim2_input @ self.dim2_weight + self.dim2_bias
        return dim1_encoder, dim2_encoder


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim1': 4, 'input_dim2': 4, 'batch_feature': 4,
        'latent_dim': 4}]
