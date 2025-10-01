from _paritybench_helpers import _mock_config
import torch


class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.
            filters_3, self.args.filters_3))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector. 
        """
        batch_size = embedding.shape[0]
        global_context = torch.mean(torch.matmul(embedding, self.
            weight_matrix), dim=1)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.matmul(embedding,
            transformed_global.view(batch_size, -1, 1)))
        representation = torch.matmul(embedding.permute(0, 2, 1),
            sigmoid_scores)
        return representation, sigmoid_scores


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(filters_3=4)}]
