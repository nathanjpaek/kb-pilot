import torch
import torch.nn.functional as F
import torch.nn as nn


class MetricLearningLoss(nn.Module):
    """
    Generic loss function to be used in a metric learning setting
    """

    def __init__(self, embedding_size, n_classes, device='cpu', *args, **kwargs
        ):
        super(MetricLearningLoss, self).__init__()
        self.embedding_size = embedding_size
        self.n_classes = n_classes
        self.device = device

    def forward(self, inputs, targets):
        raise NotImplementedError()


class AngularMarginLoss(MetricLearningLoss):
    """
    Generic angular margin loss definition
    (see https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch)

    "ElasticFace: Elastic Margin Loss for Deep Face Recognition",
    Boutros et al., https://arxiv.org/abs/2109.09416v2
    """

    def __init__(self, embedding_size, n_classes, device='cpu', scale=None,
        m1=1, m2=0, m3=0, eps=1e-06):
        super(AngularMarginLoss, self).__init__(embedding_size, n_classes,
            device=device)
        self.fc = nn.Linear(embedding_size, n_classes, bias=False)
        self.scale = scale
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Compute ArcFace loss for inputs of shape [B, E] and
        targets of size [B]

        B: batch size
        E: embedding size
        """
        self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
        inputs_norms = torch.norm(inputs, p=2, dim=1)
        normalized_inputs = inputs / inputs_norms.unsqueeze(-1).repeat(1,
            inputs.size(1))
        scales = torch.tensor([self.scale], device=inputs.device).repeat(inputs
            .size(0)) if self.scale is not None else inputs_norms
        cosines = self.fc(normalized_inputs).clamp(-1, 1)
        preds = torch.argmax(cosines, dim=1)
        angles = torch.arccos(cosines)
        numerator = scales.unsqueeze(-1) * (torch.cos(self.m1 * angles +
            self.m2) - self.m3)
        numerator = torch.diagonal(numerator.transpose(0, 1)[targets])
        excluded = torch.cat([(scales[i] * torch.cat((cosines[i, :y],
            cosines[i, y + 1:])).unsqueeze(0)) for i, y in enumerate(
            targets)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(excluded),
            dim=1)
        loss = -torch.mean(numerator - torch.log(denominator + self.eps))
        return normalized_inputs, preds, loss


def get_inputs():
    return [torch.rand([4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'embedding_size': 4, 'n_classes': 4}]
