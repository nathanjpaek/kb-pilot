import torch
import torch.nn as nn


class DeterministicSumming(nn.Module):
    """Transform a tensor into repetitions of its sum.

    Intended for use in tests, not useful for actual learning. The last
    dimension of the input should contain feature vectors. The result will be
    an array of matching shape with the last dimension replaced by repeated
    utility values (i.e. sums).

    Let's use this as a pairwise utility function. As an example, consider
    this pairing. There are two instances with two objects each. All object
    combinations are considered. Objects have two features.

    >>> import torch
    >>> pairs = torch.tensor(
    ...    [[[0.5000, 0.6000, 0.5000, 0.6000],
    ...      [0.5000, 0.6000, 1.5000, 1.6000],
    ...      [1.5000, 1.6000, 0.5000, 0.6000],
    ...      [1.5000, 1.6000, 1.5000, 1.6000]],
    ...     [[2.5000, 2.6000, 2.5000, 2.6000],
    ...      [2.5000, 2.6000, 3.5000, 3.6000],
    ...      [3.5000, 3.6000, 2.5000, 2.6000],
    ...      [3.5000, 3.6000, 3.5000, 3.6000]]])

    We can compute the mock utility of this pairing as follows:

    >>> utility = DeterministicSumming(input_size=2)
    >>> utilities = utility(pairs)
    >>> utilities
    tensor([[[ 2.2000],
             [ 4.2000],
             [ 4.2000],
             [ 6.2000]],
    <BLANKLINE>
            [[10.2000],
             [12.2000],
             [12.2000],
             [14.2000]]])

    Note that for example :math:`2.2 = 0.5 + 0.6 + 0.5 + 0.6`, that is

    >>> utilities[0][0] == pairs[0][0].sum()
    tensor([True])

    Parameters
    ----------
    input_size : int
        The size of the last dimension of the input.

    output_size : int
        The size of the last dimension of the output. Defaults to `1` to make
        it more convenient to use this as a utility.
    """

    def __init__(self, input_size: 'int', output_size: 'int'=1):
        super().__init__()
        self.output_size = output_size

    def forward(self, inputs):
        """Forward inputs through the network.

        Parameters
        ----------
        inputs : tensor
            The input tensor of shape (N, *, I), where I is the input size.

        Returns
        -------
        tensor
            A tensor of shape (N, *, O), where O is the output size.
        """
        summed = inputs.sum(dim=-1)
        repeated = summed.view(-1, 1).repeat(1, self.output_size).view(
            summed.shape + (self.output_size,))
        return repeated


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
