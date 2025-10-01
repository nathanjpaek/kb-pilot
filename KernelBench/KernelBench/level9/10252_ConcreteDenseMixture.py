import torch
import numpy as np
from torch import nn


class ConcreteDropout(nn.Module):

    def __init__(self, weight_regularizer=1e-06, dropout_regularizer=1e-05,
        init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        init_min = np.log(init_min) - np.log(1.0 - init_min)
        init_max = np.log(init_max) - np.log(1.0 - init_max)
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max)
            )

    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit)
        out = layer(self._concrete_dropout(x, p))
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1.0 - p) * torch.log(1.0 - p)
        input_dimensionality = x[0].numel()
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality
        regularization = weights_regularizer + dropout_regularizer
        return out, regularization

    def _concrete_dropout(self, x, p):
        eps = 1e-07
        temp = 0.1
        unif_noise = torch.rand_like(x)
        drop_prob = torch.log(p + eps) - torch.log(1 - p + eps) + torch.log(
            unif_noise + eps) - torch.log(1 - unif_noise + eps)
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        x = torch.mul(x, random_tensor)
        x /= retain_prob
        return x


class ConcreteDenseMixture(nn.Module):

    def __init__(self, X_dim, Y_dim, nb_features, weight_regularizer,
        dropout_regularizer, verbose=True):
        super(ConcreteDenseMixture, self).__init__()
        self.verbose = verbose
        self.rank = 2
        self.linear1 = nn.Linear(X_dim, nb_features)
        self.linear2 = nn.Linear(nb_features, nb_features)
        self.linear3 = nn.Linear(nb_features, nb_features)
        self.linear4_mu = nn.Linear(nb_features, Y_dim - 1)
        self.linear4_logvar = nn.Linear(nb_features, Y_dim - 1)
        self.linear4_F = nn.Linear(nb_features, (Y_dim - 1) * self.rank)
        self.linear4_mu2 = nn.Linear(nb_features, Y_dim - 1)
        self.linear4_logvar2 = nn.Linear(nb_features, Y_dim - 1)
        self.linear4_F2 = nn.Linear(nb_features, (Y_dim - 1) * self.rank)
        self.linear4_alpha = nn.Linear(nb_features, 1)
        self.linear4_mu_classifier = nn.Linear(nb_features, 1)
        self.linear4_logvar_classifier = nn.Linear(nb_features, 1)
        self.conc_drop1 = ConcreteDropout(weight_regularizer=
            weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conc_drop2 = ConcreteDropout(weight_regularizer=
            weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conc_drop3 = ConcreteDropout(weight_regularizer=
            weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conc_drop_mu = ConcreteDropout(weight_regularizer=
            weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conc_drop_logvar = ConcreteDropout(weight_regularizer=
            weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conc_drop_F = ConcreteDropout(weight_regularizer=
            weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conc_drop_mu2 = ConcreteDropout(weight_regularizer=
            weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conc_drop_logvar2 = ConcreteDropout(weight_regularizer=
            weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conc_drop_F2 = ConcreteDropout(weight_regularizer=
            weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conc_drop_alpha = ConcreteDropout(weight_regularizer=
            weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conc_drop_mu_classifier = ConcreteDropout(weight_regularizer=
            weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.conc_drop_logvar_classifier = ConcreteDropout(weight_regularizer
            =weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.tanh = nn.Tanh()

    def forward(self, x):
        regularization = torch.empty(12, device=x.device)
        x1, regularization[0] = self.conc_drop1(x, nn.Sequential(self.
            linear1, self.tanh))
        x2, regularization[1] = self.conc_drop2(x1, nn.Sequential(self.
            linear2, self.tanh))
        x3, regularization[2] = self.conc_drop3(x2, nn.Sequential(self.
            linear3, self.tanh))
        mean, regularization[3] = self.conc_drop_mu(x3, self.linear4_mu)
        logvar, regularization[4] = self.conc_drop_logvar(x3, self.
            linear4_logvar)
        F, regularization[5] = self.conc_drop_F(x3, self.linear4_F)
        mean2, regularization[6] = self.conc_drop_mu2(x3, self.linear4_mu2)
        logvar2, regularization[7] = self.conc_drop_logvar2(x3, self.
            linear4_logvar2)
        F2, regularization[8] = self.conc_drop_F2(x3, self.linear4_F2)
        alpha, regularization[9] = self.conc_drop_alpha(x3, self.linear4_alpha)
        mean_classifier, regularization[10] = self.conc_drop_mu_classifier(x3,
            self.linear4_mu_classifier)
        logvar_classifier, regularization[11
            ] = self.conc_drop_logvar_classifier(x3, self.
            linear4_logvar_classifier)
        return (mean, logvar, F, mean2, logvar2, F2, alpha, mean_classifier,
            logvar_classifier, regularization.sum())


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'X_dim': 4, 'Y_dim': 4, 'nb_features': 4,
        'weight_regularizer': 4, 'dropout_regularizer': 0.5}]
