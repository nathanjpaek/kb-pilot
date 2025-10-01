import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureFinder(nn.Module):

    def __init__(self):
        super(TextureFinder, self).__init__()
        self.encoder_conv1 = nn.Conv2d(in_channels=1, out_channels=4,
            kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=True
            )
        self.encoder_conv1.bias.data.zero_()
        self.encoder_conv1.weight.data[:, :, :, :] = 1 / 0.32 + torch.normal(
            mean=torch.tensor(0.0), std=torch.tensor(0.0001))
        self.encoder_normalization1 = nn.GroupNorm(1, 4, eps=1e-05, affine=True
            )
        self.encoder_conv2 = nn.Conv2d(in_channels=4, out_channels=16,
            kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=True
            )
        self.encoder_conv2.bias.data.zero_()
        self.encoder_conv2.weight.data[:, :, :, :] = 1 / (8 * 16
            ) + torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.0001))
        self.encoder_normalization2 = nn.GroupNorm(1, 16, eps=1e-05, affine
            =True)
        self.encoder_conv3 = nn.Conv2d(in_channels=16, out_channels=32,
            kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=True
            )
        self.encoder_conv3.bias.data.zero_()
        self.encoder_conv3.weight.data[:, :, :, :] = 1 / (8 * 16
            ) + torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.001))
        self.encoder_normalization3 = nn.GroupNorm(1, 32, eps=1e-05, affine
            =True)
        self.encoder_mu = nn.Conv2d(in_channels=32, out_channels=16,
            kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=True
            )
        self.encoder_mu.bias.data.zero_()
        self.encoder_mu.weight.data[:, :, :, :] = 1 / (8 * 16) + torch.normal(
            mean=torch.tensor(0.0), std=torch.tensor(0.0001))
        self.encoder_log_var = nn.Conv2d(in_channels=32, out_channels=16,
            kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=True
            )
        self.encoder_log_var.bias.data[:] = -2.3
        self.encoder_log_var.weight.data.zero_()
        self.decoder_conv1 = nn.ConvTranspose2d(16, 32, kernel_size=8,
            stride=2, padding=3)
        self.decoder_conv1.bias.data.zero_()
        self.decoder_conv1.weight.data[:, :, :, :] = 1 / (8 * 8 * 8
            ) + torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.0001))
        self.decoder_normalization1 = nn.GroupNorm(1, 32, eps=1e-05, affine
            =True)
        self.decoder_conv2 = nn.ConvTranspose2d(32, 64, kernel_size=8,
            stride=2, padding=3, output_padding=0, groups=1, bias=True,
            dilation=1)
        self.decoder_conv2.bias.data.zero_()
        self.decoder_conv2.weight.data[:, :, :, :] = 1 / (8 * 8 * 8
            ) + torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.0001))
        self.decoder_normalization2 = nn.GroupNorm(1, 64, eps=1e-05, affine
            =True)
        self.decoder_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=8,
            stride=2, padding=3, output_padding=0, groups=1, bias=True,
            dilation=1)
        self.decoder_conv3.bias.data.zero_()
        self.decoder_conv3.weight.data[:, :, :, :] = 1 / (4 * 8 * 8
            ) + torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.001))
        self.decoder_normalization3 = nn.GroupNorm(1, 32, eps=1e-05, affine
            =True)
        self.decoder_conv5 = nn.ConvTranspose2d(32, 1, kernel_size=8,
            stride=2, padding=3, output_padding=0, groups=1, bias=True,
            dilation=1)
        self.decoder_conv5.bias.data[:] = -(0.5 / 0.24)
        self.decoder_conv5.weight.data[:, :, :, :] = 1 / (32 * 8 * 8 * 0.24
            ) + torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.001))

    def forward(self, input):
        embeddings_enc0 = F.relu(self.encoder_conv1(input))
        embeddings_enc0 = self.encoder_normalization1(embeddings_enc0)
        embeddings_enc1 = F.relu(self.encoder_conv2(embeddings_enc0))
        embeddings_enc1 = self.encoder_normalization2(embeddings_enc1)
        embeddings_enc2 = F.relu(self.encoder_conv3(embeddings_enc1))
        embeddings_enc2 = self.encoder_normalization3(embeddings_enc2)
        mu = self.encoder_mu(embeddings_enc2)
        log_var = self.encoder_log_var(embeddings_enc2)
        sample = self.sample_from_mu_log_var(mu, log_var)
        embeddings_dec1 = F.relu(self.decoder_conv1(sample, output_size=
            embeddings_enc2.size()))
        embeddings_dec1 = self.decoder_normalization1(embeddings_dec1)
        embeddings_dec2 = F.relu(self.decoder_conv2(embeddings_dec1,
            output_size=embeddings_enc1.size()))
        embeddings_dec2 = self.decoder_normalization2(embeddings_dec2)
        embeddings_dec3 = F.relu(self.decoder_conv3(embeddings_dec2,
            output_size=embeddings_enc0.size()))
        embeddings_dec3 = self.decoder_normalization3(embeddings_dec3)
        reconstructed = F.sigmoid(self.decoder_conv5(embeddings_dec3,
            output_size=input.size()))
        return (reconstructed, mu, log_var, sample, embeddings_enc1,
            embeddings_dec2)

    def sample_from_mu_log_var(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + eps * std
        return sample


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
