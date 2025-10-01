import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """
    A logistic regression model of the form
    P(y = 1 | x) = 1 / (1 + exp(-(mx + b)))
    """

    def __init__(self, init_m=1.0, init_b=1.0):
        """
        Initialize a logistic regression model by defining its initial
        parameters.

        The ``nn.Parameter`` wrapper is needed if we intend to use vanilla
        ``torch.Tensors`` as parameters of this module, so that this object
        knows it has trainable parameters (e.g. when calling the
        ``.parameters()`` method on this object, or moving the model to GPU via
        ``.cuda``).

        If we assign parameters which are already subclasses of ``nn.Module``
        (e.g. the ``nn.Linear`` layer), ``nn.Parameter`` is not needed.

        Parameters
        ----------
        init_m : ``float``, optional (defualt: 1.0)
            Initial estimate for ``m``, the slope of the model
        init_b : ``float``, optional (default: 1.0)
            Initial estimate for ``b``, the y-intercept of the model
        """
        super(LogisticRegression, self).__init__()
        self.m = nn.Parameter(torch.tensor(init_m))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, x):
        """
        Forward pass through the model, which produces a prediction
        ``\\hat{y}`` for each given ``x`` in the batch.

        Parameters
        ----------
        x : ``torch.Tensor`` of shape ``(batch_size, )``
            The single input.

        Returns
        -------
        scores : ``torch.Tensor`` of shape ``(batch_size, )``
            The raw logits of the predicted output
        """
        scores = self.m * x + self.b
        return scores


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
