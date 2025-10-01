import torch
import torch.nn as nn


class TAE_decoder(nn.Module):
    """
    Class for temporal autoencoder decoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    """

    def __init__(self, n_hidden=64, pooling=8):
        super().__init__()
        self.pooling = pooling
        self.n_hidden = n_hidden
        self.up_layer = nn.Upsample(size=pooling)
        self.deconv_layer = nn.ConvTranspose1d(in_channels=self.n_hidden,
            out_channels=self.n_hidden, kernel_size=10, stride=1, padding=
            self.pooling // 2)

    def forward(self, features):
        upsampled = self.up_layer(features)
        out_deconv = self.deconv_layer(upsampled)[:, :, :self.pooling
            ].contiguous()
        out_deconv = out_deconv.view(out_deconv.shape[0], -1)
        return out_deconv


def get_inputs():
    return [torch.rand([4, 64, 4])]


def get_init_inputs():
    return [[], {}]
