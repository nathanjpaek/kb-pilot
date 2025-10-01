import torch
import torch.nn as nn


class ParallelAttention(nn.Module):

    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.ques_linear = nn.Linear(self.embedding_size, self.hidden_size)
        self.img_linear = nn.Linear(self.embedding_size, self.hidden_size)
        self.ques_linear_t = nn.Linear(self.hidden_size, 1)
        self.img_linear_t = nn.Linear(self.hidden_size, 1)
        self.affinity = nn.Linear(self.embedding_size, self.hidden_size,
            bias=False)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, ques_embed_t, img_embed):
        ques_embed = ques_embed_t.permute(0, 2, 1)
        img_embed_t = img_embed.permute(0, 2, 1)
        C = torch.matmul(self.affinity(ques_embed_t), img_embed)
        C = self.activation(C)
        C_t = C.permute(0, 2, 1)
        a = self.img_linear(img_embed_t)
        b = self.ques_linear(ques_embed_t)
        h_vis = a + torch.matmul(C_t, b)
        h_vis = self.activation(h_vis)
        h_ques = b + torch.matmul(C, a)
        h_ques = self.activation(h_ques)
        attention_vis = self.img_linear_t(h_vis).squeeze()
        attention_ques = self.ques_linear_t(h_ques).squeeze()
        attention_vis = self.softmax(attention_vis)
        attention_ques = self.softmax(attention_ques)
        attention_vis = attention_vis.unsqueeze(1)
        attention_ques = attention_ques.unsqueeze(1)
        attention_feat_vis = torch.mul(img_embed, attention_vis)
        attention_feat_vis = torch.sum(attention_feat_vis, dim=-1)
        attention_feat_ques = torch.mul(ques_embed, attention_ques)
        attention_feat_ques = torch.sum(attention_feat_ques, dim=-1)
        return attention_feat_vis, attention_feat_ques


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'embedding_size': 4, 'hidden_size': 4}]
