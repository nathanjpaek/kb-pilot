import torch
import torch.nn as nn


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, is_rnn=False, mode='fan_in', nonlinearity=
    'leaky_relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        if is_rnn:
            for name, param in module.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, bias)
                elif 'weight' in name:
                    nn.init.kaiming_uniform_(param, a=a, mode=mode,
                        nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode,
                nonlinearity=nonlinearity)
    elif is_rnn:
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, bias)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param, a=a, mode=mode, nonlinearity
                    =nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity
            =nonlinearity)
    if not is_rnn and hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            kaiming_init(m)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            constant_init(m, 1)
        elif isinstance(m, nn.Linear):
            xavier_init(m)
        elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
            kaiming_init(m, is_rnn=True)


class PVABlock(nn.Module):

    def __init__(self, num_steps, in_channels, embedding_channels=512,
        inner_channels=512):
        super(PVABlock, self).__init__()
        self.num_steps = num_steps
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.embedding_channels = embedding_channels
        self.order_embeddings = nn.Parameter(torch.randn(self.num_steps,
            self.embedding_channels), requires_grad=True)
        self.v_linear = nn.Linear(self.in_channels, self.inner_channels,
            bias=False)
        self.o_linear = nn.Linear(self.embedding_channels, self.
            inner_channels, bias=False)
        self.e_linear = nn.Linear(self.inner_channels, 1, bias=False)
        init_weights(self.modules())

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        o_out = self.o_linear(self.order_embeddings).view(1, self.num_steps,
            1, self.inner_channels)
        v_out = self.v_linear(x).unsqueeze(1)
        att = self.e_linear(torch.tanh(o_out + v_out)).squeeze(3)
        att = torch.softmax(att, dim=2)
        out = torch.bmm(att, x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_steps': 4, 'in_channels': 4}]
