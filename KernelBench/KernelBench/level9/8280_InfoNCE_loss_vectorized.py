import torch
import torch.nn as nn


class InfoNCE_loss_vectorized(nn.Module):
    """
        SimCLR loss: https://github.com/google-research/simclr // https://github.com/sthalles/SimCLR
    """

    def __init__(self, temperature):
        super(InfoNCE_loss_vectorized, self).__init__()
        self.temperature = temperature
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, graph_out, sent_out):
        total_loss = 0
        for g, s in zip(graph_out, sent_out):
            similarities = self.cos(g, s)
            similarities = similarities / self.temperature
            exp_tensor = torch.exp(similarities)
            loss = exp_tensor[0] / torch.sum(exp_tensor)
            loss = -torch.log(loss)
            total_loss = total_loss + loss
        total_loss_final = total_loss / len(graph_out)
        return total_loss_final


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'temperature': 4}]
