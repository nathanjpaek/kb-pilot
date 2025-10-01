from torch.nn import Module
import math
import torch
import numpy as np
from torch.nn import *
from torch.optim import AdamW
from typing import Union


class FactorizationMachine(Module):
    """
    [Factorization Machine Recommendation Model]
    Learns latent space features to characterize similarity of dataset features
    to compute a recommendation as a function of dataset features. Dataset
    features can be mixed / hybrid such that you can combine information
    on both the recommended object and the recommendation target to generate
    an informed similarity or recommendation / ranking metric.
    """

    def __init__(self, data_dim, hidden_dim=25, seed=None) ->None:
        """
        Instantiate class attributes for FM. Constructs a feature similarity matrix
        F of shape (x_features, hidden_dim) to learn implicit representations of
        all trainable features in the data for recommendation or ranking.
        :param data_dim <int>:      Number of features to learn from in the dataset.
        :param hidden_dim <int>:    Dimension of the latent space of features.
        :param seed <int>:          Random seed fixture for reproducibility.
        """
        super().__init__()
        self.input_dim = data_dim
        self.hidden_dim = hidden_dim
        self.torch_gen = None
        if seed is not None:
            self.torch_gen = torch.manual_seed(seed)
        """ Matrix Factorization """
        self.F = Parameter(torch.empty((self.input_dim, self.hidden_dim)),
            requires_grad=True)
        init.xavier_uniform_(self.F)
        """ Linear Regression """
        self.V = Parameter(torch.empty((self.input_dim, 1)), requires_grad=True
            )
        init.xavier_uniform_(self.V)
        self.bias = Parameter(torch.zeros(1), requires_grad=True)
        """ Gaussian Regression """
        self.gaussian_dist = Linear(in_features=self.hidden_dim, out_features=2
            )

    def forward(self, x: 'torch.Tensor'):
        """
        Compute FactorizationMachine(x). Returns a mean and standard deviation for the recommendation.
        :param x <torch.Tensor>:    Factorization machine input Tensor of shape (N, input_dim).
        """
        sq_sm = torch.matmul(x, self.F) ** 2
        sm_sq = torch.matmul(x ** 2, self.F ** 2)
        lin_reg = torch.matmul(x, self.V)
        latent = self.bias + lin_reg + 0.5 * sq_sm - sm_sq
        output = self.gaussian_dist(latent)
        return output[:, 0], torch.abs(output[:, 1])

    def fit(self, X: 'Union[torch.Tensor, np.ndarray]', Y:
        'Union[torch.Tensor, np.ndarray]', mask:
        'Union[torch.Tensor, np.ndarray]'=None, cycles=100, lr=0.002,
        batch_frac=0.01, regularize=0.01, patience=3, verbose=False):
        """
        Train the Factorization Machine.
        :param X <torch.Tensor>:        Input training data features of shape (N, X).
        :param Y <torch.Tensor>:        Target training data class / score vector of shape (N, 1).
        :param mask <torch.Tensor>:     Feature observability mask for X of shape (N, X).
        :param cycles <int>:            Number of gradient descent cycles.
        :param lr <float>:              Learning rate. Re-calibrated to order of values in matrix M.
        :param batch_frac <float>:      Fraction of the dataset to set as the batch size.
        :param regularize <float>:      Weight decay lambda for regularization in AdamW.
        :param patience <int>:          Number of cycles of convergence before termination.
        :param verbose <bool>:          Output training progress information.
        """
        if any([len(X.shape) != 2, len(Y.shape) != 2, mask is not None and 
            mask.shape != X.shape, X.shape[1] != self.input_dim, Y.shape[1] !=
            1, cycles <= 0, lr <= 0, batch_frac <= 0, regularize < 0]):
            None
            return
        N = X.shape[0]
        if not torch.is_tensor(X):
            X = torch.Tensor(X)
        if not torch.is_tensor(Y):
            Y = torch.Tensor(Y)
        mask_tensor = torch.ones(X.shape)
        if mask is not None:
            mask_tensor = torch.where(torch.Tensor(mask) != 0, 1, 0)
        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=regularize)
        model_opt = dict(self.state_dict())
        loss_opt = float('inf')
        timer = 0
        for i in range(cycles):
            for _ in range(math.ceil(1 / batch_frac)):
                rand_idx = torch.randint(N, size=(math.ceil(batch_frac * N)
                    ,), generator=self.torch_gen)
                X_batch = X[rand_idx]
                Y_batch = Y[rand_idx]
                mask_batch = mask_tensor[rand_idx]
                self.zero_grad()
                Y_mu, Y_sigma = self(X_batch * mask_batch)
                loss = GaussianNLLLoss()(Y_mu, Y_batch, Y_sigma)
                loss.sum().backward()
                optimizer.step()
            if i % math.ceil(cycles / 5) == 0 and verbose:
                None
            if loss.sum().item() < loss_opt:
                model_opt = dict(self.state_dict())
                loss_opt = loss.sum().item()
                timer = 0
            else:
                timer += 1
                if timer > patience:
                    self.load_state_dict(model_opt)
                    break


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'data_dim': 4}]
