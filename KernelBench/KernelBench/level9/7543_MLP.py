import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, indim, hs, outdim, mlp_drop):
        super().__init__()
        """
        eh, et, |eh-et|, eh*et
        """
        indim = 4 * indim
        self.linear1 = nn.Linear(indim, 2 * hs)
        self.linear2 = nn.Linear(2 * hs, outdim)
        self.drop = nn.Dropout(mlp_drop)

    def forward(self, head_rep, tail_rep):
        """
        :param head_rep: (?, hs)
        :param tail_rep: (?, hs)
        :param doc_rep: (1, hs)
        :return: logits (?, outdim)
        """
        mlp_input = [head_rep, tail_rep, torch.abs(head_rep - tail_rep), 
            head_rep * tail_rep]
        mlp_input = torch.cat(mlp_input, -1)
        h = self.drop(F.relu(self.linear1(mlp_input)))
        return self.linear2(h)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'indim': 4, 'hs': 4, 'outdim': 4, 'mlp_drop': 0.5}]
