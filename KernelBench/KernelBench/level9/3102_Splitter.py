import torch
import numpy as np


class Splitter(torch.nn.Module):
    """
    An implementation of "Splitter: Learning Node Representations that Capture Multiple Social Contexts" (WWW 2019).
    Paper: http://epasto.org/papers/www2019splitter.pdf
    """

    def __init__(self, dimensions, lambd, base_node_count, node_count, device):
        """
        Splitter set up.
        :param dimensions: Dimension of embedding vectors
        :param lambd: Parameter that determine how much personas spread from original embedding
        :param base_node_count: Number of nodes in the source graph.
        :param node_count: Number of nodes in the persona graph.
        :param device: Device which torch use
        """
        super(Splitter, self).__init__()
        self.dimensions = dimensions
        self.lambd = lambd
        self.base_node_count = base_node_count
        self.node_count = node_count
        self.device = device

    def create_weights(self):
        """
        Creating weights for embedding.
        """
        self.base_node_embedding = torch.nn.Embedding(self.base_node_count,
            self.dimensions, padding_idx=0)
        self.node_embedding = torch.nn.Embedding(self.node_count, self.
            dimensions, padding_idx=0)

    def initialize_weights(self, base_node_embedding, mapping, str2idx):
        """
        Using the base embedding and the persona mapping for initializing the embedding matrices.
        :param base_node_embedding: Node embedding of the source graph.
        :param mapping: Mapping of personas to nodes.
        :param str2idx: Mapping string of original network to index in original network
        """
        persona_embedding = np.array([base_node_embedding[str2idx[
            original_node]] for node, original_node in mapping.items()])
        self.node_embedding.weight.data = torch.nn.Parameter(torch.Tensor(
            persona_embedding))
        self.base_node_embedding.weight.data = torch.nn.Parameter(torch.
            Tensor(base_node_embedding), requires_grad=False)

    def calculate_main_loss(self, node_f, feature_f, targets):
        """
        Calculating the main loss which is used to learning based on persona random walkers
        It will be act likes centrifugal force from the base embedding
        :param node_f: Embedding vectors of source nodes
        :param feature_f: Embedding vectors of target nodes to predict
        :param targets: Boolean vector whether negative samples or not
        """
        node_f = torch.nn.functional.normalize(node_f, p=2, dim=1)
        feature_f = torch.nn.functional.normalize(feature_f, p=2, dim=1)
        scores = torch.sum(node_f * feature_f, dim=1)
        scores = torch.sigmoid(scores)
        main_loss = targets * torch.log(scores) + (1 - targets) * torch.log(
            1 - scores)
        main_loss = -torch.mean(main_loss)
        return main_loss

    def calculate_regularization(self, source_f, original_f):
        """
         Calculating the main loss which is used to learning based on persona random walkers
         It will be act likes centripetal force from the base embedding
         :param source_f: Embedding vectors of source nodes
         :param original_f: Embedding vectors of base embedding of source nodes
         """
        source_f = torch.nn.functional.normalize(source_f, p=2, dim=1)
        original_f = torch.nn.functional.normalize(original_f, p=2, dim=1)
        scores = torch.sum(source_f * original_f, dim=1)
        scores = torch.sigmoid(scores)
        regularization_loss = -torch.mean(torch.log(scores))
        return regularization_loss

    def forward(self, node_f, feature_f, targets, source_f, original_f):
        """
        1.main loss part
        :param node_f: Embedding vectors of source nodes
        :param feature_f: Embedding vectors of target nodes to predict
        :param targets: Boolean vector whether negative samples or not

        2.regularization part
        :param source_f: Embedding vectors of source nodes
        :param original_f: Embedding vectors of base embedding of source nodes
        """
        main_loss = self.calculate_main_loss(node_f, feature_f, targets)
        regularization_loss = self.calculate_regularization(source_f,
            original_f)
        loss = main_loss + self.lambd * regularization_loss
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dimensions': 4, 'lambd': 4, 'base_node_count': 4,
        'node_count': 4, 'device': 0}]
