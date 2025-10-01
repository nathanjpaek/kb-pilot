import torch
import torch.nn as nn


def init_drop(dropout):
    if dropout > 0:
        return nn.Dropout(dropout)
    else:
        return lambda x: x


class SelfAttention(nn.Module):

    def __init__(self, hidden_dim, attn_drop, txt):
        """
        Description
        -----------
        This part is used to calculate type-level attention and semantic-level attention, and utilize them to generate :math:`z^{sc}` and :math:`z^{mp}`.

        .. math::
           w_{n}&=\\frac{1}{|V|}\\sum\\limits_{i\\in V} \\textbf{a}^\\top \\cdot \\tanh\\left(\\textbf{W}h_i^{n}+\\textbf{b}\\right) \\\\
           \\beta_{n}&=\\frac{\\exp\\left(w_{n}\\right)}{\\sum_{i=1}^M\\exp\\left(w_{i}\\right)} \\\\
           z &= \\sum_{n=1}^M \\beta_{n}\\cdot h^{n}

        Parameters
        ----------
        txt : str
            A str to identify view, MP or SC

        Returns
        -------
        z : matrix
            The fused embedding matrix

        """
        super(SelfAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)),
            requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        self.softmax = nn.Softmax(dim=0)
        self.attn_drop = init_drop(attn_drop)
        self.txt = txt

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        None
        z = 0
        for i in range(len(embeds)):
            z += embeds[i] * beta[i]
        return z


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4, 'attn_drop': 0.5, 'txt': 4}]
