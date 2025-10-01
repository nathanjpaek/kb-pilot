import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m,
        nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


class EnsembleFC(nn.Module):

    def __init__(self, in_features: 'int', out_features: 'int',
        ensemble_size: 'int', weight_decay: 'float'=0.0, bias: 'bool'=True
        ) ->None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features,
            out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        pass

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])

    def extra_repr(self) ->str:
        return 'in_features={}, out_features={}, bias={}'.format(self.
            in_features, self.out_features, self.bias is not None)


class EnsembleModel(nn.Module):

    def __init__(self, feature_size, ensemble_size, use_decay=False):
        super(EnsembleModel, self).__init__()
        self.nn1 = EnsembleFC(feature_size + feature_size, feature_size,
            ensemble_size, weight_decay=2.5e-05)
        self.use_decay = use_decay
        self.apply(weights_init_)
        self.swish = Swish()

    def forward(self, state_latent, action_latent):
        x = torch.cat([state_latent, action_latent], 2)
        nn1_output = self.nn1(x)
        return nn1_output

    def get_decay_loss(self):
        decay_loss = 0.0
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)
                    ) / 2.0
        return decay_loss

    def loss(self, mean, labels):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(labels.shape) == 3
        mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
        total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_size': 4, 'ensemble_size': 4}]
