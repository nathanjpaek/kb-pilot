import torch
import torch.nn as nn


class AttendedTextEncoding(nn.Module):

    def __init__(self, hidden_size):
        super(AttendedTextEncoding, self).__init__()
        self.sentence_linear = nn.Linear(hidden_size, hidden_size)
        self.att_linear1 = nn.Linear(hidden_size * 2, hidden_size // 2)
        self.att_linear2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, cap_emb, sentence_encode, mask=None):
        """

        :param cap_emb: batch * sentence_len * hidden size
        :param sentence_encode: batch * hidden size
        :param mask: batch * sentence_len
        :return: batch * hidden size
        """
        sentence = torch.relu(self.sentence_linear(sentence_encode))
        fusion_emb = torch.cat([cap_emb, sentence[:, None, :].expand(
            cap_emb.shape)], dim=2)
        att = torch.relu(self.att_linear1(fusion_emb))
        att = self.att_linear2(att)
        if mask is not None:
            att = att.masked_fill(~mask.bool(), float('-inf'))
        att = torch.softmax(att.transpose(1, 2), dim=2)
        attended_emb = (att @ cap_emb).squeeze(1)
        return attended_emb


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
