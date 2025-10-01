import torch
from torch import nn
from sklearn.cluster import KMeans


class translatedSigmoid(nn.Module):

    def __init__(self):
        super(translatedSigmoid, self).__init__()
        self.beta = nn.Parameter(torch.tensor([-3.5]))

    def forward(self, x):
        beta = torch.nn.functional.softplus(self.beta)
        alpha = -beta * 6.9077542789816375
        return torch.sigmoid((x + alpha) / beta)


class DistNet(nn.Module):

    def __init__(self, dim, num_points):
        super().__init__()
        self.num_points = num_points
        self.points = nn.Parameter(torch.randn(num_points, dim),
            requires_grad=False)
        self.trans = translatedSigmoid()
        self.initialized = False

    def __dist2__(self, x):
        t1 = (x ** 2).sum(-1, keepdim=True)
        t2 = torch.transpose((self.points ** 2).sum(-1, keepdim=True), -1, -2)
        t3 = 2.0 * torch.matmul(x, torch.transpose(self.points, -1, -2))
        return (t1 + t2 - t3).clamp(min=0.0)

    def forward(self, x):
        with torch.no_grad():
            D2 = self.__dist2__(x)
            min_d = D2.min(dim=-1)[0]
            return self.trans(min_d)

    def kmeans_initializer(self, embeddings):
        km = KMeans(n_clusters=self.num_points).fit(embeddings)
        self.points.data = torch.tensor(km.cluster_centers_, device=self.
            points.device)
        self.initialized = True


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'num_points': 4}]
