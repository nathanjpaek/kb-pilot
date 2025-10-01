import torch


class ProtoLoss(torch.nn.Module):

    def __init__(self, num_classes, num_support, num_queries, ndim):
        super(ProtoLoss, self).__init__()
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_queries = num_queries
        self.ndim = ndim

    def euclidean_distance(self, a, b):
        N, _D = a.shape[0], a.shape[1]
        M = b.shape[0]
        a = torch.repeat_interleave(a.unsqueeze(1), repeats=M, dim=1)
        b = torch.repeat_interleave(b.unsqueeze(0), repeats=N, dim=0)
        return 1.0 * torch.sum(torch.pow(a - b, 2), 2)

    def forward(self, x, q, labels_onehot):
        protox = torch.mean(1.0 * x.reshape([self.num_classes, self.
            num_support, self.ndim]), 1)
        dists = self.euclidean_distance(protox, q)
        logpy = torch.log_softmax(-1.0 * dists, 0).transpose(1, 0).view(self
            .num_classes, self.num_queries, self.num_classes)
        ce_loss = -1.0 * torch.mean(torch.mean(logpy * labels_onehot.float(
            ), 1))
        accuracy = torch.mean((torch.argmax(labels_onehot.float(), -1).
            float() == torch.argmax(logpy, -1).float()).float())
        return ce_loss, accuracy


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4,
        4, 4])]


def get_init_inputs():
    return [[], {'num_classes': 4, 'num_support': 4, 'num_queries': 4,
        'ndim': 4}]
