import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda


class SequenceQuantizerSoftEMA(nn.Module):

    def __init__(self, codebook_size, d_model, l1_cost=1000, entropy_cost=
        5e-05, num_samples=10, temp=1.0, epsilon=1e-05, padding_idx=None):
        super(SequenceQuantizerSoftEMA, self).__init__()
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.padding_idx = padding_idx
        self.codebook = nn.Parameter(torch.FloatTensor(self.codebook_size,
            self.d_model), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.codebook)
        self.l1_cost = l1_cost
        self.entropy_cost = entropy_cost
        self.num_samples = num_samples
        self.temp = temp
        self._epsilon = epsilon

    def entropy(self, tensor):
        return torch.mean(torch.sum(-1 * torch.matmul(F.log_softmax(tensor,
            dim=1), tensor.t()), dim=1))

    def forward(self, inputs, l1_cost=None, entropy_cost=None, temp=None):
        if l1_cost is None:
            l1_cost = self.l1_cost
        if entropy_cost is None:
            entropy_cost = self.entropy_cost
        if temp is None:
            temp = self.temp
        input_shape = inputs.size()
        flat_input = inputs.reshape(-1, self.d_model)
        norm_C = self.codebook / self.codebook.norm(2, dim=1)[:, None]
        flat_input = flat_input / flat_input.norm(2, dim=1)[:, None]
        distances = F.softmax(torch.matmul(flat_input, norm_C.t()), dim=1)
        reconstruction = torch.matmul(distances, norm_C).view(input_shape)
        l1_loss = nn.L1Loss()
        loss = l1_cost * l1_loss(distances, torch.zeros_like(distances)
            ) + entropy_cost * self.entropy(distances)
        return reconstruction, loss

    def cluster(self, inputs):
        input_shape = inputs.size()
        inputs.dim()
        flat_input = inputs.reshape(-1, self.d_model)
        flat_input = flat_input / flat_input.norm(2, dim=1)[:, None]
        codebook = self.codebook / self.codebook.norm(2, dim=1)[:, None]
        distances = F.softmax(torch.matmul(flat_input, codebook.t()).
            reshape(-1, self.output_nheads, codebook.shape[0]), dim=2)
        reconstruction = torch.matmul(distances, codebook).view(input_shape)
        encoding_indices = torch.argmax(distances, dim=1).reshape(-1, self.
            output_nheads)
        return reconstruction, encoding_indices, distances

    def set_codebook(self, new_codebook):
        self.codebook.copy_(new_codebook)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'codebook_size': 4, 'd_model': 4}]
