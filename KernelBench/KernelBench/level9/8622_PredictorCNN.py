import torch
import torch.nn as nn


class PredictorCNN(nn.Module):

    def __init__(self, latent_dim=1024, reduced_dim=64):
        super(PredictorCNN, self).__init__()
        self.latent_dim = latent_dim
        self.reduced_dim = reduced_dim
        self.conv1 = nn.Conv2d(self.latent_dim, self.reduced_dim, 1, bias=False
            )
        self.conv2 = nn.Conv2d(self.latent_dim, self.reduced_dim, 1, bias=False
            )

    def forward(self, latents, context):
        reduced_latents = self.conv1(latents)
        prediction = self.conv2(context)
        return reduced_latents, prediction

    def train_order_block_ids(self):
        return [[0, 1]]


def get_inputs():
    return [torch.rand([4, 1024, 64, 64]), torch.rand([4, 1024, 64, 64])]


def get_init_inputs():
    return [[], {}]
