import random
import torch
import numpy as np
from torch import nn


class MLP(nn.Module):

    def __init__(self, kernels, num_features, num_hiddens, normalize=True,
        num_updates=3000, batch_size=128, weight_decay=0.0001, soft_preds=False
        ):
        super().__init__()
        self.kernels = kernels
        num_kernels = len(kernels)
        self.linear_1 = nn.Linear(num_features, num_hiddens)
        self.act = nn.Tanh()
        self.linear_2 = nn.Linear(num_hiddens, num_kernels)
        self.softmax = nn.LogSoftmax(dim=1)
        self.mean = None
        self.std = None
        self._normalize = normalize
        self.num_updates = num_updates
        self.batch_size = batch_size
        self.soft_preds = soft_preds
        self.weight_decay = weight_decay

    def forward(self, x):
        y1 = self.linear_1.forward(x)
        y = self.act.forward(y1)
        y = self.linear_2.forward(y)
        return self.softmax.forward(y)

    def normalize(self, X):
        if self._normalize:
            return (X - self.mean) / self.std
        return X

    def predict_proba(self, x):
        x = self.normalize(x)
        tx = torch.from_numpy(x).float()
        y = self.forward(tx)
        return np.exp(y.detach().numpy())

    def predict(self, x):
        y = self.predict_proba(x)
        return y.argmax(axis=1)

    def fit(self, X, y):
        if self._normalize:
            self.mean = X.mean(axis=0, keepdims=True)
            self.std = X.std(axis=0, keepdims=True)
            self.std[self.std < 0.0001] = 0.0001
            X = self.normalize(X)
        updates = 0
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001,
            weight_decay=self.weight_decay)
        loss = torch.nn.KLDivLoss(reduction='batchmean'
            ) if self.soft_preds else torch.nn.NLLLoss()
        indices = list(range(X.shape[0]))
        num_batches = len(indices) // self.batch_size
        prev_loss = None
        num_iter_no_impr = 0
        while updates < self.num_updates:
            random.shuffle(indices)
            total_loss = 0
            batches_seen = 0
            for bnum in range(num_batches):
                bb = self.batch_size * bnum
                be = bb + self.batch_size
                Xb = X[indices[bb:be]]
                yb = y[indices[bb:be]]
                tx = torch.from_numpy(Xb).float()
                if self.soft_preds:
                    ty = torch.from_numpy(yb).float()
                else:
                    ty = torch.from_numpy(yb).long()
                optimizer.zero_grad()
                z = self.forward(tx)
                loss_val = loss(z, ty)
                loss_val.backward()
                optimizer.step()
                sloss = loss_val.detach().numpy()
                total_loss += sloss
                updates += 1
                batches_seen += 1
                if updates > self.num_updates:
                    break
            total_loss /= batches_seen
            if prev_loss is not None:
                impr = (prev_loss - total_loss) / prev_loss
                if impr < 0.0001:
                    num_iter_no_impr += 1
                else:
                    num_iter_no_impr = 0
            prev_loss = total_loss
            if num_iter_no_impr > 4:
                break


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernels': [4, 4], 'num_features': 4, 'num_hiddens': 4}]
