import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureSegmentation(nn.Module):

    def __init__(self):
        super(TextureSegmentation, self).__init__()
        self.decoder_conv1 = nn.ConvTranspose2d(16, 32, kernel_size=(8, 16),
            stride=2, padding=(3, 7))
        self.decoder_conv1.bias.data.zero_()
        self.decoder_conv1.weight.data[:, :, :, :] = 1 / (8 * 8 * 8
            ) + torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.0001))
        self.decoder_normalization1 = nn.GroupNorm(1, 32, eps=1e-05, affine
            =True)
        self.decoder_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=(8, 16),
            stride=2, padding=(3, 7), output_padding=0, groups=1, bias=True,
            dilation=1)
        self.decoder_conv2.bias.data.zero_()
        self.decoder_conv2.weight.data[:, :, :, :] = 1 / (8 * 8 * 8
            ) + torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.0001))
        self.decoder_normalization2 = nn.GroupNorm(1, 16, eps=1e-05, affine
            =True)
        self.decoder_conv3 = nn.ConvTranspose2d(16, 8, kernel_size=(8, 16),
            stride=2, padding=(3, 7), output_padding=0, groups=1, bias=True,
            dilation=1)
        self.decoder_conv3.bias.data.zero_()
        self.decoder_conv3.weight.data[:, :, :, :] = 1 / (4 * 8 * 8
            ) + torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.001))
        self.decoder_normalization3 = nn.GroupNorm(1, 8, eps=1e-05, affine=True
            )
        self.decoder_conv5 = nn.ConvTranspose2d(8, 1, kernel_size=(8, 16),
            stride=2, padding=(3, 7), output_padding=0, groups=1, bias=True,
            dilation=1)
        self.decoder_conv5.bias.data[:] = -(0.5 / 0.24)
        self.decoder_conv5.weight.data[:, :, :, :] = 1 / (8 * 8 * 8 * 0.24
            ) + torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.001))

    def forward(self, sample):
        embeddings_dec1 = F.relu(self.decoder_conv1(sample, output_size=
            torch.empty(sample.size()[0], 32, sample.size()[2] * 2, sample.
            size()[3] * 2).size()))
        embeddings_dec1 = self.decoder_normalization1(embeddings_dec1)
        embeddings_dec2 = F.relu(self.decoder_conv2(embeddings_dec1,
            output_size=torch.empty(embeddings_dec1.size()[0], 16, 
            embeddings_dec1.size()[2] * 2, embeddings_dec1.size()[3] * 2).
            size()))
        embeddings_dec2 = self.decoder_normalization2(embeddings_dec2)
        embeddings_dec3 = F.relu(self.decoder_conv3(embeddings_dec2,
            output_size=torch.empty(embeddings_dec2.size()[0], 8, 
            embeddings_dec2.size()[2] * 2, embeddings_dec2.size()[3] * 2).
            size()))
        embeddings_dec3 = self.decoder_normalization3(embeddings_dec3)
        segment = F.sigmoid(self.decoder_conv5(embeddings_dec3, output_size
            =torch.empty(embeddings_dec3.size()[0], 1, embeddings_dec3.size
            ()[2] * 2, embeddings_dec3.size()[3] * 2).size()))
        return segment


def get_inputs():
    return [torch.rand([4, 16, 4, 4])]


def get_init_inputs():
    return [[], {}]
