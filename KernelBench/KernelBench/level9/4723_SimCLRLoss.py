import torch
import numpy as np
import torch.nn as nn
import torch.utils.model_zoo


class SimCLRLoss(nn.Module):

    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.T = temperature
        self.ce = nn.CrossEntropyLoss()
        self.norm = nn.functional.normalize
        self.softmax = nn.functional.softmax
        self.cosine = nn.CosineSimilarity(dim=-1)
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye(2 * batch_size, 2 * batch_size, k=-batch_size)
        l2 = np.eye(2 * batch_size, 2 * batch_size, k=batch_size)
        mask = diag + l1 + l2
        return mask

    def forward(self, f1, f2):
        batch_size = f1.shape[0]
        sim_matrix = self.cosine(f1.unsqueeze(1), f2.unsqueeze(0)) / self.T
        label = torch.arange(0, batch_size, device=sim_matrix.device)
        loss = self.ce(sim_matrix, label) + self.ce(sim_matrix.t(), label)
        return loss * 0.5


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'temperature': 4}]
