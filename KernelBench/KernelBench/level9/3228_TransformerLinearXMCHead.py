import torch
import numpy as np
import torch.nn as nn


class TransformerLinearXMCHead(nn.Module):
    """XMC head for Transformers

    Containing label weight embeddings and label bias embeddings
    """

    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.label_pad = num_labels
        self.num_labels = num_labels
        self.W = nn.Embedding(num_labels + 1, hidden_size, padding_idx=self
            .label_pad)
        self.b = nn.Embedding(num_labels + 1, 1, padding_idx=self.label_pad)
        self.random_init()

    @property
    def device(self):
        return self.W.weight.device

    def random_init(self):
        """Initialize the weight and bias embeddings

        Initialize label weight embedding with N(0, 0.02) while keeping PAD
        column to be 0. Initialize label bias embedding with 0.
        """
        mat = 0.02 * np.random.randn(self.label_pad, self.W.weight.shape[1])
        mat = np.hstack([mat, np.zeros([mat.shape[0], 1])])
        self.init_from(mat)

    def inherit(self, prev_head, C):
        prev_W = prev_head.W.weight[:-1, :].detach().numpy()
        prev_b = prev_head.b.weight[:-1, :].detach().numpy()
        cur_W = C * prev_W
        cur_b = C * prev_b
        mat = np.hstack([cur_W, cur_b])
        self.init_from(mat)

    def bootstrap(self, prob, **kwargs):
        """Initialize head with weights learned from linear model using transformer embeddings

        Args:
            prob (MLProblem): the multi-label problem to bootstrap with
            kwargs:
                Cp (float): the weight on positive samples. Default 100.0
                Cn (float): the weight on negative samples. Default 100.0
                threshold (float): the threshold to sparsify the model
        """
        Cp = kwargs.get('Cp', 100.0)
        Cn = kwargs.get('Cn', 100.0)
        threshold = kwargs.get('threshold', 0)
        mat = MLModel.train(prob, threshold=threshold, Cp=Cp, Cn=Cn)
        mat = mat.W.toarray().T
        self.init_from(mat)

    def init_from(self, mat):
        """Initialize the weight and bias embeddings with given matrix

        Args:
            mat (ndarray): matrix used for initialize, shape = (nr_labels, hidden_size + 1)
        """
        if not isinstance(mat, np.ndarray):
            raise ValueError('Expect ndarray to initialize label embedding')
        if mat.shape[0] != self.label_pad:
            raise ValueError('nr_labels mismatch!')
        mat = np.vstack([mat, np.zeros([1, mat.shape[1]])])
        self.W = nn.Embedding.from_pretrained(torch.FloatTensor(mat[:, :-1]
            ), freeze=False, sparse=True, padding_idx=self.label_pad)
        self.b = nn.Embedding.from_pretrained(torch.FloatTensor(mat[:, -1])
            .view((self.label_pad + 1, 1)), freeze=False, sparse=True,
            padding_idx=self.label_pad)

    def forward(self, pooled_output=None, output_indices=None, num_device=1):
        if output_indices is None:
            W_act = self.W.weight[:-1, :].repeat(num_device, 1, 1)
            b_act = self.b.weight[:-1].repeat(num_device, 1, 1)
        else:
            output_indices = output_indices
            W_act = self.W(output_indices)
            b_act = self.b(output_indices)
        return W_act, b_act


def get_inputs():
    return []


def get_init_inputs():
    return [[], {'hidden_size': 4, 'num_labels': 4}]
