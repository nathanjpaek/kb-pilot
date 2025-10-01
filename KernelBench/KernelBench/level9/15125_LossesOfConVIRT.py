import torch
import torch.nn as nn


class LossesOfConVIRT(nn.Module):
    """

    """

    def __init__(self, tau=0.1, lambd=0.75):
        super(LossesOfConVIRT, self).__init__()
        self.tau = tau
        self.lambd = lambd

    def tmp_loss(self, v, u, index):
        """

        """
        assert v.size(0) == u.size(0)
        item1 = torch.exp(torch.divide(torch.cosine_similarity(v[index], u[
            index], dim=0), self.tau))
        item2 = torch.exp(torch.divide(torch.cosine_similarity(v[index].
            unsqueeze(0), u, dim=1), self.tau)).sum()
        loss = -torch.log(torch.divide(item1, item2))
        return loss

    def image_text(self, v, u, index):
        """

        """
        assert v.size(0) == u.size(0)
        cos = torch.nn.CosineSimilarity(dim=0)
        item1 = torch.exp(torch.divide(cos(v[index], u[index]), self.tau))
        cos2 = torch.nn.CosineSimilarity(dim=1)
        item2 = torch.exp(torch.divide(cos2(v[index].unsqueeze(0), u), self
            .tau)).sum()
        loss = -torch.log(torch.divide(item1, item2))
        return loss

    def text_image(self, v, u, index):
        """

        """
        assert v.size(0) == u.size(0)
        cos = torch.nn.CosineSimilarity(dim=0)
        item1 = torch.exp(torch.divide(cos(v[index], u[index]), self.tau))
        cos2 = torch.nn.CosineSimilarity(dim=1)
        item2 = torch.exp(torch.divide(cos2(v, u[index].unsqueeze(0)), self
            .tau)).sum()
        loss = -torch.log(torch.divide(item1, item2)).item()
        return loss

    def forward(self, v, u):
        """

        :return:
        """
        assert v.size(0) == u.size(0)
        res = 0.0
        v = v.float()
        u = u.float()
        for i in range(v.size(0)):
            res += self.lambd * self.image_text(v, u, i) + (1 - self.lambd
                ) * self.text_image(v, u, i)
        res /= v.size(0)
        return res


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
