import torch
import torch.nn as nn
import torch.nn.init


class CRF_S(nn.Module):
    """Conditional Random Field (CRF) layer. This version is used in Lample et al. 2016, has less parameters than CRF_L.

    args:
        hidden_dim: input dim size
        tagset_size: target_set_size
        if_biase: whether allow bias in linear trans

    """

    def __init__(self, hidden_dim, tagset_size, if_bias=True):
        super(CRF_S, self).__init__()
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=if_bias)
        self.transitions = nn.Parameter(torch.Tensor(self.tagset_size, self
            .tagset_size))

    def rand_init(self):
        """random initialization
        """
        utils.init_linear(self.hidden2tag)
        self.transitions.data.zero_()

    def forward(self, feats):
        """
        args:
            feats (batch_size, seq_len, hidden_dim) : input score from previous layers
        return:
            output from crf layer ( (batch_size * seq_len), tag_size, tag_size)
        """
        scores = self.hidden2tag(feats).view(-1, self.tagset_size, 1)
        ins_num = scores.size(0)
        crf_scores = scores.expand(ins_num, self.tagset_size, self.tagset_size
            ) + self.transitions.view(1, self.tagset_size, self.tagset_size
            ).expand(ins_num, self.tagset_size, self.tagset_size)
        return crf_scores


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4, 'tagset_size': 4}]
