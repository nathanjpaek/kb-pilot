import torch
import torch.nn as nn
import torch.nn.init


class CRF(nn.Module):
    """
    Conditional Random Field Module

    Parameters
    ----------
    hidden_dim : ``int``, required.
        the dimension of the input features.
    tagset_size : ``int``, required.
        the size of the target labels.
    if_bias: ``bool``, optional, (default=True).
        whether the linear transformation has the bias term.
    """

    def __init__(self, hidden_dim: 'int', tagset_size: 'int', if_bias:
        'bool'=True):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=if_bias)
        self.transitions = nn.Parameter(torch.Tensor(self.tagset_size, self
            .tagset_size))

    def rand_init(self):
        """
        random initialization
        """
        utils.init_linear(self.hidden2tag)
        self.transitions.data.zero_()

    def forward(self, feats):
        """
        calculate the potential score for the conditional random field.

        Parameters
        ----------
        feats: ``torch.FloatTensor``, required.
            the input features for the conditional random field, of shape (*, hidden_dim).

        Returns
        -------
        output: ``torch.FloatTensor``.
            A float tensor of shape (ins_num, from_tag_size, to_tag_size)
        """
        scores = self.hidden2tag(feats).view(-1, 1, self.tagset_size)
        ins_num = scores.size(0)
        crf_scores = scores.expand(ins_num, self.tagset_size, self.tagset_size
            ) + self.transitions.view(1, self.tagset_size, self.tagset_size
            ).expand(ins_num, self.tagset_size, self.tagset_size)
        return crf_scores


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4, 'tagset_size': 4}]
