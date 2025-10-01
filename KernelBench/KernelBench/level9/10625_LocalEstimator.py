import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalEstimator(nn.Module):

    def __init__(self, input_size):
        super(LocalEstimator, self).__init__()
        self.input2hid = nn.Linear(input_size, 64)
        self.hid2hid = nn.Linear(64, 32)
        self.hid2out = nn.Linear(32, 1)

    def forward(self, sptm_s):
        hidden = F.leaky_relu(self.input2hid(sptm_s))
        hidden = F.leaky_relu(self.hid2hid(hidden))
        out = self.hid2out(hidden)
        return out

    def eval_on_batch(self, pred, lens, label, mean, std):
        label = nn.utils.rnn.pack_padded_sequence(label, lens, batch_first=True
            )[0]
        label = label.view(-1, 1)
        label = label * std + mean
        pred = pred * std + mean
        loss = torch.abs(pred - label) / (label + EPS)
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
