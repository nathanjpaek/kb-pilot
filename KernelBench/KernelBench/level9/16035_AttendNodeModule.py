import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class AttendNodeModule(nn.Module):

    def forward(self, node_vectors, query):
        """
        Args:
            node_vectors [Tensor] (num_node, dim_v) : node feature vectors
            query [Tensor] (dim_v, ) : query vector
        Returns:
            attn [Tensor] (num_node, ): node attention by *SOFTMAX*. Actually it is attribute value attention, because we regard attribute values as nodes
        """
        logit = torch.matmul(node_vectors, query)
        attn = F.softmax(logit, dim=0)
        return attn


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
