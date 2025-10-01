import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention_ElementWiseProduct(nn.Module):
    """
      Input:
          behavior: 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
          candidate: 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output:
          attention_weight: 3D tensor with shape: ``(batch_size, field_size, 1)``.
    """

    def __init__(self, embedding_size):
        super().__init__()
        self.linear1 = nn.Linear(4 * embedding_size, 32)
        self.linear2 = nn.Linear(32, 1)
        self.prelu = nn.PReLU()

    def forward(self, behavior, candidate):
        candidate = candidate.expand_as(behavior)
        embed_input = torch.cat([behavior, candidate, behavior - candidate,
            behavior * candidate], dim=2)
        output = self.prelu(self.linear1(embed_input))
        output = F.sigmoid(self.linear2(output))
        return output


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'embedding_size': 4}]
