import torch
import numpy as np
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()
        self._hidden_size = hidden_size
        self._recurrent = recurrent
        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            N = hxs.size(0)
            T = int(x.size(0) / N)
            x = x.view(T, N, x.size(1))
            masks = masks.view(T, N)
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu(
                )
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()
            has_zeros = [0] + has_zeros + [T]
            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                rnn_scores, hxs = self.gru(x[start_idx:end_idx], hxs *
                    masks[start_idx].view(1, -1, 1))
                outputs.append(rnn_scores)
            x = torch.cat(outputs, dim=0)
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)
        return x, hxs


class OscBase(NNBase):

    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(OscBase, self).__init__(recurrent, num_inputs, hidden_size)

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_
                (x, 0), np.sqrt(2))
        self.time_idx = num_inputs // 2 - 1
        self.osc_fanout1 = nn.Linear(1, 12)
        self.osc_fanout2 = nn.Linear(12, 12)
        self.layer1 = nn.Linear(num_inputs, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size * 2)
        self.layerA1 = nn.Linear(hidden_size * 2 + 12, hidden_size)
        self.layerA2 = nn.Linear(hidden_size, hidden_size)
        self.layerC1 = nn.Linear(hidden_size * 2 + 12, hidden_size)
        self.layerC2 = init_(nn.Linear(hidden_size, 1))
        self.train()

    def forward(self, inputs):
        x = inputs
        phase = torch.sin(0.004125 * 2 * 3.14159 / 0.4 * x[:, self.time_idx])
        phase = phase.unsqueeze(1)
        x[:, self.time_idx] = 0
        x[:, -1] = 0
        o = torch.tanh(self.osc_fanout1(phase))
        o = torch.tanh(self.osc_fanout2(o))
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        xo = torch.cat((x, o), 1)
        x_a = torch.tanh(self.layerA1(xo))
        x_a = torch.tanh(self.layerA2(x_a))
        x_c = torch.tanh(self.layerC1(xo))
        x_c = self.layerC2(x_c)
        return x_c, x_a


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_inputs': 4}]
