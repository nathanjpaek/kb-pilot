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


class EntityClassifier(nn.Module):

    def __init__(self, hs, num_class, mlp_drop):
        super().__init__()
        indim = 2 * hs
        self.classifier = MLP(indim, hs, num_class, mlp_drop)

    def forward(self, global_head, global_tail, local_head, local_tail,
        path2ins):
        ins2path = torch.transpose(path2ins, 0, 1)
        global_head = torch.matmul(path2ins, global_head)
        global_tail = torch.matmul(path2ins, global_tail)
        head_rep, tail_rep = [], []
        head_rep.append(local_head)
        tail_rep.append(local_tail)
        head_rep.append(global_head)
        tail_rep.append(global_tail)
        head_rep = torch.cat(head_rep, dim=-1)
        tail_rep = torch.cat(tail_rep, dim=-1)
        pred = self.classifier(head_rep, tail_rep)
        pred = pred.squeeze(-1)
        pred = torch.sigmoid(pred)
        pred = pred.unsqueeze(0)
        ins2path = ins2path.unsqueeze(-1)
        pred = torch.max(pred * ins2path, dim=1)[0]
        return pred


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hs': 4, 'num_class': 4, 'mlp_drop': 0.5}]
