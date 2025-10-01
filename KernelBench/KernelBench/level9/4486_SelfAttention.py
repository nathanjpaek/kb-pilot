import torch
from torch.nn import init
from torch.nn.parameter import Parameter


class SelfAttention(torch.nn.Module):

    def __init__(self, wv_dim: 'int', maxlen: 'int'):
        super(SelfAttention, self).__init__()
        self.wv_dim = wv_dim
        self.maxlen = maxlen
        self.M = Parameter(torch.empty(size=(wv_dim, wv_dim)))
        init.kaiming_uniform_(self.M.data)
        self.attention_softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_embeddings):
        mean_embedding = torch.mean(input_embeddings, (1,)).unsqueeze(2)
        product_1 = torch.matmul(self.M, mean_embedding)
        product_2 = torch.matmul(input_embeddings, product_1).squeeze(2)
        results = self.attention_softmax(product_2)
        return results

    def extra_repr(self):
        return 'wv_dim={}, maxlen={}'.format(self.wv_dim, self.maxlen)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'wv_dim': 4, 'maxlen': 4}]
