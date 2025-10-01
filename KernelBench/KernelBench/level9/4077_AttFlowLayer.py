import torch
import torch.nn as nn
import torch.nn.functional as F


class AttFlowLayer(nn.Module):

    def __init__(self, embed_length):
        super(AttFlowLayer, self).__init__()
        self.embed_length = embed_length
        self.alpha = nn.Linear(3 * embed_length, 1, bias=False)

    def forward(self, context, query):
        batch_size = context.shape[0]
        query = query.unsqueeze(0).expand((batch_size, query.shape[0], self
            .embed_length))
        shape = batch_size, context.shape[1], query.shape[1], self.embed_length
        context_extended = context.unsqueeze(2).expand(shape)
        query_extended = query.unsqueeze(1).expand(shape)
        multiplied = torch.mul(context_extended, query_extended)
        cated = torch.cat((context_extended, query_extended, multiplied), 3)
        S = self.alpha(cated).view(batch_size, context.shape[1], query.shape[1]
            )
        S_softmax_row = F.softmax(S, dim=1).permute(0, 2, 1)
        F.softmax(S, dim=2)
        query_masks = torch.sign(torch.abs(torch.sum(query, dim=-1)))
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, context.
            size()[1])
        S_softmax_row = S_softmax_row * query_masks
        S_softmax_row_1 = S_softmax_row.unsqueeze(3).expand(S_softmax_row.
            shape[0], S_softmax_row.shape[1], S_softmax_row.shape[2], self.
            embed_length)
        context_1 = context_extended.permute(0, 2, 1, 3)
        attd = torch.mul(S_softmax_row_1, context_1)
        G = torch.sum(attd, 1)
        H = torch.sum(attd, 2)
        G = torch.cat((context, G), 2)
        return G, H


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'embed_length': 4}]
