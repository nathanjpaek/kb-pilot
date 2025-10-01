import torch
import torch.nn as nn


class AspectMean(nn.Module):

    def __init__(self, max_sen_len):
        """
        :param max_sen_len: maximum length of sentence
        """
        super(AspectMean, self).__init__()
        self.max_sen_len = max_sen_len

    def forward(self, aspect):
        """

        :param aspect: size: [batch_size, max_asp_len, embed_size]
        :return: aspect mean embedding, size: [batch_size, max_sen_len, embed_size]
        """
        len_tmp = torch.sum(aspect != 0, dim=2)
        aspect_len = torch.sum(len_tmp != 0, dim=1).unsqueeze(dim=1).float()
        out = aspect.sum(dim=1)
        out = out.div(aspect_len).unsqueeze(dim=1).expand(-1, self.
            max_sen_len, -1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'max_sen_len': 4}]
