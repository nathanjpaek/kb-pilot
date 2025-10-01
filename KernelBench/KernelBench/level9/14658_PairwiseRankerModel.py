import torch
import torch.onnx
import torch.nn as nn


class PairwiseRankerModel(nn.Module):

    def __init__(self, embedding_size):
        super(PairwiseRankerModel, self).__init__()
        self.query_doc_transform = torch.nn.Linear(in_features=
            embedding_size * 2, out_features=embedding_size)
        self.compare_transform = torch.nn.Linear(in_features=embedding_size *
            2, out_features=1)

    def forward(self, query_embedding, doc_1_embedding, doc_2_embedding):
        query_doc_1_rep = torch.cat((query_embedding, doc_1_embedding), 1)
        query_doc_1_rep = torch.sigmoid(self.query_doc_transform(
            query_doc_1_rep))
        query_doc_2_rep = torch.cat((query_embedding, doc_2_embedding), 1)
        query_doc_2_rep = torch.sigmoid(self.query_doc_transform(
            query_doc_2_rep))
        compare = torch.cat((query_doc_1_rep, query_doc_2_rep), 1)
        compare = self.compare_transform(compare)
        return torch.sigmoid(compare)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'embedding_size': 4}]
