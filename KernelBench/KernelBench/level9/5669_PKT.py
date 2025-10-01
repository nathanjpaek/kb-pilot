import torch
import torch.nn as nn
import torch.optim


class PKT(nn.Module):
    """Probabilistic Knowledge Transfer for deep representation learning
    Code from author: https://github.com/passalis/probabilistic_kt"""

    def __init__(self):
        super(PKT, self).__init__()

    def forward(self, f_s, f_t):
        return self.cosine_similarity_loss(f_s, f_t)

    @staticmethod
    def cosine_similarity_loss(output_net, target_net, eps=1e-07):
        output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1,
            keepdim=True))
        output_net = output_net / (output_net_norm + eps)
        output_net[output_net != output_net] = 0
        target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1,
            keepdim=True))
        target_net = target_net / (target_net_norm + eps)
        target_net[target_net != target_net] = 0
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0
        model_similarity = model_similarity / torch.sum(model_similarity,
            dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity,
            dim=1, keepdim=True)
        loss = torch.mean(target_similarity * torch.log((target_similarity +
            eps) / (model_similarity + eps)))
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
