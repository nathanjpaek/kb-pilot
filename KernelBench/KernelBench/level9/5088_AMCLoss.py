import torch
import torch.nn as nn
import torch.nn.functional as F


class AMCLoss(nn.Module):

    def __init__(self, in_features, out_features, s=None, m=None, device='cuda'
        ):
        """
        Angular Margin Contrastive Loss

        https://arxiv.org/pdf/2004.09805.pdf
        
        Code converted over from Tensorflow to Pytorch

        """
        super(AMCLoss, self).__init__()
        self.m = 0.5 if not m else m
        self.s = 1.0 if not s else s
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.device = device

    def forward(self, X, labels=None):
        """
        input shape (N, in_features)
        """
        X = F.normalize(X, p=2, dim=1)
        batch_size = X.shape[0]
        wf = self.fc(X)
        half = int(batch_size / 2)
        _, target_hard = torch.max(F.softmax(wf, dim=1), 1)
        try:
            neighbor_bool = torch.eq(target_hard[:half], target_hard[half:])
            inner = torch.sum(X[:half] * X[half:], axis=1)
        except:
            neighbor_bool = torch.eq(target_hard[:half + 1], target_hard[half:]
                )
            inner = torch.sum(X[:half + 1] * X[half:], axis=1)
        geo_desic = torch.acos(torch.clamp(inner, -1e-07, 1e-07)) * self.s
        geo_losses = torch.where(neighbor_bool, torch.square(geo_desic),
            torch.square(F.relu(self.m - geo_desic))).clamp(min=1e-12)
        return torch.mean(geo_losses), wf


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
