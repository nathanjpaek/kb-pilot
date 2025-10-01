import torch
import torch.nn as nn


class SelfCriticCriterion(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, props, s_words, tgt, advantage):
        advantage = (advantage - advantage.mean()) / advantage.std().clamp(min
            =1e-08)
        s_props = props.gather(2, s_words.unsqueeze(2)).squeeze()
        mask = (tgt > 0).float()
        advantage = advantage.unsqueeze(1).repeat(1, mask.size(1))
        advantage = advantage.detach()
        return -(s_props * mask * advantage).sum() / mask.sum()


def get_inputs():
    return [torch.ones([4, 4, 4], dtype=torch.int64), torch.ones([4, 4],
        dtype=torch.int64), torch.rand([4, 4]), torch.rand([4])]


def get_init_inputs():
    return [[], {}]
