import torch
from torch import nn
import torch.nn.functional as F


class AttentionGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(AttentionGRUCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size=input_size + num_embeddings,
            hidden_size=hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = torch.unsqueeze(self.h2h(prev_hidden), dim=1)
        res = torch.add(batch_H_proj, prev_hidden_proj)
        res = torch.tanh(res)
        e = self.score(res)
        alpha = F.softmax(e, dim=1)
        alpha = alpha.permute(0, 2, 1)
        context = torch.squeeze(torch.bmm(alpha, batch_H), dim=1)
        concat_context = torch.cat([context, char_onehots], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha


class AttentionHead(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(AttentionHead, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels
        self.attention_cell = AttentionGRUCell(in_channels, hidden_size,
            out_channels, use_gru=False)
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char.long(), onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None, batch_max_length=25):
        batch_size = inputs.shape[0]
        num_steps = batch_max_length
        hidden = torch.zeros((batch_size, self.hidden_size)).type_as(inputs)
        output_hiddens = torch.zeros((batch_size, num_steps, self.hidden_size)
            ).type_as(inputs)
        if targets is not None:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets[:, i],
                    onehot_dim=self.num_classes)
                hidden, _alpha = self.attention_cell(hidden, inputs,
                    char_onehots)
                output_hiddens[:, i, :] = hidden[0]
            probs = self.generator(output_hiddens)
        else:
            targets = torch.zeros(batch_size).int().type_as(inputs)
            probs = None
            char_onehots = None
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=
                    self.num_classes)
                hidden, _alpha = self.attention_cell(hidden, inputs,
                    char_onehots)
                probs_step = self.generator(hidden)
                if probs is None:
                    probs = torch.unsqueeze(probs_step, dim=1)
                else:
                    probs = torch.cat([probs, torch.unsqueeze(probs_step,
                        dim=1)], dim=1)
                next_input = probs_step.argmax(dim=1)
                targets = next_input
        return probs


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'hidden_size': 4}]
