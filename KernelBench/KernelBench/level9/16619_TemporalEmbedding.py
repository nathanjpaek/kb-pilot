import math
import torch
import torch.nn as nn


class FixedEmbedding(nn.Module):

    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(
            10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):

    def __init__(self, d_model, embed_type='fixed', data='ETTh'):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if data == 'ETTm':
            self.minute_embed = Embed(minute_size, d_model)
        elif data == 'SolarEnergy':
            minute_size = 6
            self.minute_embed = Embed(minute_size, d_model)
        elif data == 'WADI':
            minute_size = 60
            second_size = 6
            self.minute_embed = Embed(minute_size, d_model)
            self.second_emebd = Embed(second_size, d_model)
        elif data == 'SMAP':
            minute_size = 60
            second_size = 15
            self.minute_embed = Embed(minute_size, d_model)
            self.second_emebd = Embed(second_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        second_x = self.second_emebd(x[:, :, 5]) if hasattr(self,
            'second_embed') else 0.0
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self,
            'minute_embed') else 0.0
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x + second_x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
