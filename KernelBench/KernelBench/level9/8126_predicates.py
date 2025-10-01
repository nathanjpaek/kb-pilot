import torch
import torch.nn as nn
import torch.nn.functional as func


class predicates(nn.Module):

    def __init__(self, num_predicates, body_len):
        """
        Use these to express a choice amongst predicates. For use when
        learning rules.

        Parameters:
        ----------

        num_predicates: The domain size of predicates
        body_len: The number of predicates to choose
        """
        super().__init__()
        self.log_weights = nn.Parameter(torch.zeros(body_len,
            num_predicates).uniform_(-0.1, 0.1))

    def forward(self, x):
        """
        Forward function computes the attention weights and returns the result of mixing predicates.

        Parameters:
        ----------

        x: a 2D tensor whose number of columns should equal self.num_predicates

        Returns: A 2D tensor with 1 column
        -------
        """
        weights = self.get_params()
        ret = func.linear(x, weights)
        return ret

    def get_params(self):
        ret = func.softmax(self.log_weights, dim=1)
        return ret


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_predicates': 4, 'body_len': 4}]
