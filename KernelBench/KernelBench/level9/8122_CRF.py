import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F


class CRF(nn.Module):
    """
    Conditional Random Field Module

    Parameters
    ----------
    hidden_dim : ``int``, required.
        the dimension of the input features.
    tagset_size : ``int``, required.
        the size of the target labels.
    a_num : ``int``, required.
        the number of annotators.
    task : ``str``, required.
        the model task
    if_bias: ``bool``, optional, (default=True).
        whether the linear transformation has the bias term.
    """

    def __init__(self, hidden_dim: 'int', tagset_size: 'int', a_num: 'int',
        task: 'str', if_bias: 'bool'=True):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.a_num = a_num
        self.task = task
        if 'maMulVecCrowd' in self.task:
            self.maMulVecCrowd = nn.Parameter(torch.Tensor(self.a_num, self
                .tagset_size))
        if 'maAddVecCrowd' in self.task:
            self.maAddVecCrowd = nn.Parameter(torch.Tensor(self.a_num, self
                .tagset_size))
        if 'maCatVecCrowd' in self.task:
            self.maCatVecCrowd = nn.Parameter(torch.Tensor(self.a_num, self
                .tagset_size))
            self.maCatVecCrowd_latent = nn.Parameter(torch.Tensor(self.
                tagset_size))
        if 'maMulMatCrowd' in self.task:
            self.maMulMatCrowd = nn.Parameter(torch.Tensor(self.a_num, self
                .tagset_size, self.tagset_size))
        if 'maMulCRFCrowd' in self.task:
            self.maMulCRFCrowd = nn.Parameter(torch.Tensor(self.a_num, self
                .tagset_size, self.tagset_size))
        if 'maMulScoreCrowd' in self.task:
            self.maMulScoreCrowd = nn.Parameter(torch.Tensor(self.a_num,
                self.tagset_size, self.tagset_size))
        if 'maCatVecCrowd' in self.task:
            self.hidden2tag = nn.Linear(hidden_dim + self.tagset_size, self
                .tagset_size, bias=if_bias)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=
                if_bias)
        if not ('maMulCRFCrowd' in self.task and 'latent' not in self.task):
            self.transitions = nn.Parameter(torch.Tensor(self.tagset_size,
                self.tagset_size))

    def rand_init(self):
        """
        random initialization
        """
        if 'maMulVecCrowd' in self.task:
            self.maMulVecCrowd.data.fill_(1)
        if 'maAddVecCrowd' in self.task:
            self.maAddVecCrowd.data.fill_(0)
        if 'maCatVecCrowd' in self.task:
            self.maCatVecCrowd.data.fill_(0)
            self.maCatVecCrowd_latent.data.fill_(0)
        if 'maMulMatCrowd' in self.task:
            for i in range(self.a_num):
                nn.init.eye_(self.maMulMatCrowd[i])
        if 'maMulCRFCrowd' in self.task:
            for i in range(self.a_num):
                nn.init.eye_(self.maMulCRFCrowd[i])
        if 'maMulScoreCrowd' in self.task:
            for i in range(self.a_num):
                nn.init.eye_(self.maMulScoreCrowd[i])
        utils.init_linear(self.hidden2tag)
        if not ('maMulCRFCrowd' in self.task and 'latent' not in self.task):
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
        seq_len, batch_size, hid_dim = feats.shape
        ins_num = seq_len * batch_size
        self.a_num * ins_num
        if 'maCatVecCrowd' in self.task:
            feats_expand = feats.expand(self.a_num, seq_len, batch_size,
                hid_dim)
            crowd_expand = self.maCatVecCrowd.unsqueeze(1).unsqueeze(2).expand(
                self.a_num, seq_len, batch_size, self.tagset_size)
            feats_cat = torch.cat([feats_expand, crowd_expand], 3)
            scores = self.hidden2tag(feats_cat).view(self.a_num, ins_num, 1,
                self.tagset_size)
        else:
            scores = self.hidden2tag(feats).view(1, ins_num, 1, self.
                tagset_size).expand(self.a_num, ins_num, 1, self.tagset_size)
        if 'maMulVecCrowd' in self.task:
            crowd_expand = self.maMulVecCrowd.unsqueeze(1).unsqueeze(2).expand(
                self.a_num, ins_num, 1, self.tagset_size)
            scores = torch.mul(scores, crowd_expand)
        if 'maAddVecCrowd' in self.task:
            crowd_expand = self.maAddVecCrowd.unsqueeze(1).unsqueeze(2).expand(
                self.a_num, ins_num, 1, self.tagset_size)
            scores = scores + crowd_expand
        if 'maMulMatCrowd' in self.task:
            crowd = F.log_softmax(self.maMulMatCrowd, dim=2)
            crowd = crowd.view(self.a_num, 1, self.tagset_size, self.
                tagset_size).expand(self.a_num, ins_num, self.tagset_size,
                self.tagset_size)
            scores = torch.matmul(scores, crowd)
        if 'maMulCRFCrowd' in self.task and 'latent' in self.task:
            transitions = self.transitions.view(1, 1, self.tagset_size,
                self.tagset_size)
            transitions = torch.matmul(transitions, self.maMulCRFCrowd
                ).transpose(0, 1).contiguous()
        elif 'maMulCRFCrowd' in self.task:
            transitions = self.maMulCRFCrowd.view(self.a_num, 1, self.
                tagset_size, self.tagset_size).expand(self.a_num, ins_num,
                self.tagset_size, self.tagset_size)
        else:
            transitions = self.transitions.view(1, 1, self.tagset_size,
                self.tagset_size).expand(self.a_num, ins_num, self.
                tagset_size, self.tagset_size)
        scores = scores.expand(self.a_num, ins_num, self.tagset_size, self.
            tagset_size)
        crf_scores = scores + transitions
        if 'maMulScoreCrowd' in self.task:
            crowd = self.maMulScoreCrowd.view(self.a_num, 1, self.
                tagset_size, self.tagset_size).expand(self.a_num, ins_num,
                self.tagset_size, self.tagset_size)
            crf_scores = torch.matmul(crf_scores, crowd).view(self.a_num,
                ins_num, self.tagset_size, self.tagset_size)
        return crf_scores

    def latent_forward(self, feats):
        """
        ignoring crowd components
        """
        seq_len, batch_size, _hid_dim = feats.shape
        if 'maCatVecCrowd' in self.task:
            crowd_zero = self.maCatVecCrowd_latent.view(1, 1, self.tagset_size
                ).expand(seq_len, batch_size, self.tagset_size)
            feats = torch.cat([feats, crowd_zero], 2)
        scores = self.hidden2tag(feats).view(-1, 1, self.tagset_size)
        ins_num = scores.size(0)
        crf_scores = scores.expand(ins_num, self.tagset_size, self.tagset_size
            ) + self.transitions.view(1, self.tagset_size, self.tagset_size
            ).expand(ins_num, self.tagset_size, self.tagset_size)
        return crf_scores


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4, 'tagset_size': 4, 'a_num': 4, 'task': [4, 4]}
        ]
