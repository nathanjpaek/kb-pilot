import math
import torch
import torch.nn as nn


class GraphEmbedding(nn.Module):

    def __init__(self, input_size, ebd_size, use_cuda=True, use_sdne=True,
        add_noise=False, is_training=True):
        super(GraphEmbedding, self).__init__()
        self.use_cuda = use_cuda
        self.use_sdne = use_sdne
        self.add_noise = add_noise
        self.is_training = is_training
        ebd_size_1 = ebd_size * 4 if use_sdne else ebd_size
        ebd_size_2 = ebd_size * 2
        ebd_size_3 = ebd_size
        self.embedding_1 = nn.Parameter(torch.FloatTensor(input_size,
            ebd_size_1))
        if self.use_sdne:
            self.embedding_2 = nn.Parameter(torch.FloatTensor(ebd_size_1,
                ebd_size_2))
            self.embedding_3 = nn.Parameter(torch.FloatTensor(ebd_size_2,
                ebd_size_3))
        self.embedding_1.data.uniform_(-(1.0 / math.sqrt(ebd_size_1)), 1.0 /
            math.sqrt(ebd_size_1))
        if self.use_sdne:
            self.embedding_2.data.uniform_(-(1.0 / math.sqrt(ebd_size_2)), 
                1.0 / math.sqrt(ebd_size_2))
            self.embedding_3.data.uniform_(-(1.0 / math.sqrt(ebd_size_3)), 
                1.0 / math.sqrt(ebd_size_3))

    def forward(self, inputs):
        """
        :param inputs: tensor [batch, 2, seq_len]
        :return: embedded: tensor [batch, seq_len, embedding_size]
            Embed each node in the graph to a 128-dimension space
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(2)
        embedding_1 = self.embedding_1.repeat(batch_size, 1, 1)
        if self.use_sdne:
            embedding_2 = self.embedding_2.repeat(batch_size, 1, 1)
            embedding_3 = self.embedding_3.repeat(batch_size, 1, 1)
        embedded = []
        inputs = inputs.unsqueeze(1)
        for i in range(seq_len):
            embedding = torch.bmm(inputs[:, :, :, i].float(), embedding_1)
            if self.use_sdne:
                embedding = torch.bmm(embedding.float(), embedding_2)
                embedding = torch.bmm(embedding.float(), embedding_3)
            embedded.append(embedding)
        embedded = torch.cat(tuple(embedded), 1)
        return embedded


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'ebd_size': 4}]
