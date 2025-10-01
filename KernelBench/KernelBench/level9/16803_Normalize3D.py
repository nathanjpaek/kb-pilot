import torch
import torch.nn as nn


class Normalize3D(nn.Module):
    """ 
    Scale Spectrogram to be between 0 and 1
    """

    def __init__(self):
        super(Normalize3D, self).__init__()

    def forward(self, X: 'torch.Tensor'):
        if len(X.shape) != 3:
            raise ValueError(
                'Input should be 3D: [batch_size X num_features X num_steps]')
        batch_size, num_features, num_steps = X.shape
        X = X.contiguous().view(batch_size, num_features * num_steps)
        max_value = torch.max(X, dim=1)[0].detach()
        min_value = torch.min(X, dim=1)[0].detach()
        max_value = torch.unsqueeze(max_value, 1)
        min_value = torch.unsqueeze(min_value, 1)
        X = (X - min_value) / (max_value - min_value + 1e-10)
        return X.view(batch_size, num_features, num_steps)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
