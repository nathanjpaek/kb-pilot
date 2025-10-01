import torch
import torch.nn as nn
import torch.nn.functional as F


class BoF_Pooling(nn.Module):

    def __init__(self, n_codewords, features, spatial_level=0, **kwargs):
        super(BoF_Pooling, self).__init__()
        """
        Initializes a BoF Pooling layer
        :param n_codewords: the number of the codewords to be used
        :param spatial_level: 0 -> no spatial pooling, 1 -> spatial pooling at level 1 (4 regions). Note that the
         codebook is shared between the different spatial regions
        :param kwargs:
        """
        self.N_k = n_codewords
        self.spatial_level = spatial_level
        self.V, self.sigmas = None, None
        self.relu = nn.ReLU()
        self.init(features)
        self.softmax = nn.Softmax(dim=1)

    def init(self, features):
        self.V = nn.Parameter(nn.init.uniform_(torch.empty((self.N_k,
            features, 1, 1), requires_grad=True)))
        self.sigmas = nn.Parameter(nn.init.constant_(torch.empty((1, self.
            N_k, 1, 1), requires_grad=True), 0.1))

    def forward(self, input):
        x_square = torch.sum(input=input, dim=1, keepdim=True)
        y_square = torch.sum(self.V ** 2, dim=1, keepdim=True).permute([3, 
            0, 1, 2])
        dists = x_square + y_square - 2 * F.conv2d(input, self.V)
        dists = self.relu(dists)
        quantized_features = self.softmax(-dists / self.sigmas ** 2)
        if self.spatial_level == 0:
            histogram = torch.mean(quantized_features, dim=[2, 3])
        elif self.spatial_level == 1:
            shape = quantized_features.shape
            mid_1 = shape[2] / 2
            mid_1 = int(mid_1)
            mid_2 = shape[3] / 2
            mid_2 = int(mid_2)
            histogram1 = torch.mean(quantized_features[:, :, :mid_1, :mid_2
                ], [2, 3])
            histogram2 = torch.mean(quantized_features[:, :, mid_1:, :mid_2
                ], [2, 3])
            histogram3 = torch.mean(quantized_features[:, :, :mid_1, mid_2:
                ], [2, 3])
            histogram4 = torch.mean(quantized_features[:, :, mid_1:, mid_2:
                ], [2, 3])
            histogram = torch.stack([histogram1, histogram2, histogram3,
                histogram4], 1)
            histogram = torch.reshape(histogram, (-1, 4 * self.N_k))
        else:
            assert False
        return histogram * self.N_k

    def compute_output_shape(self, input_shape):
        if self.spatial_level == 0:
            return input_shape[0], self.N_k
        elif self.spatial_level == 1:
            return input_shape[0], 4 * self.N_k


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_codewords': 4, 'features': 4}]
