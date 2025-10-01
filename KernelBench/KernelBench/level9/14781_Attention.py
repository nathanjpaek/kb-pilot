import torch
import torch.nn as nn
import torch.nn


class Attention(nn.Module):

    def __init__(self, dim_i, dim_o):
        """
		build the target-aware attention
		input schema:
			dim_i: the dimension of the input feature vector
			dim_o: the dimension of the output feature vector
		output schema:
			return a aggregated vector from the context k, v of q 
		"""
        super(Attention, self).__init__()
        self.Q = nn.Linear(dim_i, dim_o)
        self.K = nn.Linear(dim_i, dim_o)
        self.V = nn.Linear(dim_i, dim_o)

    def forward(self, hist_seq_emb, hist_seq_mask, cand_emb):
        q, k, v = self.Q(cand_emb), self.K(hist_seq_emb), self.V(hist_seq_emb)
        logits = torch.sum(q.unsqueeze(1) * k, dim=2)
        logits = logits * hist_seq_mask + logits * (1 - hist_seq_mask
            ) * -2 ** 32.0
        scores = torch.softmax(logits, dim=1)
        output = torch.sum(scores.unsqueeze(2) * v, dim=1)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_i': 4, 'dim_o': 4}]
