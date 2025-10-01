import torch
from torch import nn as nn
import torch.utils.data.distributed


class SelfAttention(nn.Module):
    """
       Self SelfAttention Layer
       Given $X\\in \\mathbb{R}^{n 	imes in_feature}$, the attention is calculated by: $a=Softmax(W_2tanh(W_1X))$, where
       $W_1 \\in \\mathbb{R}^{hidden 	imes in_feature}$, $W_2 \\in \\mathbb{R}^{out_feature 	imes hidden}$.
       The final output is: $out=aX$, which is unrelated with input $n$.
    """

    def __init__(self, *, hidden, in_feature, out_feature):
        """
        The init function.
        :param hidden: the hidden dimension, can be viewed as the number of experts.
        :param in_feature: the input feature dimension.
        :param out_feature: the output feature dimension.
        """
        super(SelfAttention, self).__init__()
        self.w1 = torch.nn.Parameter(torch.FloatTensor(hidden, in_feature))
        self.w2 = torch.nn.Parameter(torch.FloatTensor(out_feature, hidden))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Use xavier_normal method to initialize parameters.
        """
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)

    def forward(self, X):
        """
        The forward function.
        :param X: The input feature map. $X \\in \\mathbb{R}^{n 	imes in_feature}$.
        :return: The final embeddings and attention matrix.
        """
        x = torch.tanh(torch.matmul(self.w1, X.transpose(1, 0)))
        x = torch.matmul(self.w2, x)
        attn = torch.nn.functional.softmax(x, dim=-1)
        x = torch.matmul(attn, X)
        return x, attn


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden': 4, 'in_feature': 4, 'out_feature': 4}]
