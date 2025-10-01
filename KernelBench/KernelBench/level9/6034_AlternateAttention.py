import torch
import torch.nn as nn


class AlternateAttention(nn.Module):

    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.x_linear = nn.Linear(self.embedding_size, self.hidden_size)
        self.g_linear = nn.Linear(self.embedding_size, self.hidden_size)
        self.linear_t = nn.Linear(self.hidden_size, 1)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, ques_embed_t, img_embed):
        img_embed_t = img_embed.permute(0, 2, 1)
        left = self.x_linear(ques_embed_t)
        H = self.activation(left)
        res = self.linear_t(H)
        a = self.softmax(res)
        a = torch.mul(ques_embed_t, a)
        a = torch.sum(a, dim=1)
        a = a.squeeze()
        left = self.x_linear(img_embed_t)
        right = self.g_linear(a)
        right = right.unsqueeze(1)
        H = self.activation(left + right)
        res = self.linear_t(H)
        a = self.softmax(res)
        a = torch.mul(img_embed_t, a)
        a = torch.sum(a, dim=1)
        attention_feat_vis = a.squeeze()
        left = self.x_linear(ques_embed_t)
        right = self.g_linear(attention_feat_vis)
        right = right.unsqueeze(1)
        H = self.activation(left + right)
        res = self.linear_t(H)
        a = self.softmax(res)
        a = torch.mul(ques_embed_t, a)
        a = torch.sum(a, dim=1)
        attention_feat_ques = a.squeeze()
        return attention_feat_vis, attention_feat_ques


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'embedding_size': 4, 'hidden_size': 4}]
