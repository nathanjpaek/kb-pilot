import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from torch.utils.data import Dataset


def compute_auc(labels, scores, pos_label=1):
    fpr, tpr, _thresholds = metrics.roc_curve(labels, scores, pos_label=
        pos_label)
    return metrics.auc(fpr, tpr)


class Subset(Dataset):

    def __init__(self, data, labels, normalize=False):
        self.ims = data
        self.labels = labels
        self.normalize = normalize
        if normalize:
            self.T = transforms.Normalize(0.5, 0.5)
        else:
            self.T = lambda x: x

    def __getitem__(self, idx):
        ret = {'ims': self.T(self.ims[idx]), 'labels': self.labels[idx]}
        return ret

    def __len__(self):
        return self.labels.shape[0]


class AlphaClassifier(nn.Module):

    def __init__(self):
        super(AlphaClassifier, self).__init__()
        self.alpha_params = nn.Parameter(torch.Tensor(np.ones(4)))
        self.scaler = StandardScaler()

    def get_alpha(self):
        return F.softmax(self.alpha_params, dim=0)

    def forward(self, x):
        z = torch.Tensor(x)
        z = z * self.get_alpha()
        z = z.sum(1)
        return z

    def predict_prob(self, x):
        return torch.sigmoid(self(x))

    def predict(self, x):
        return torch.round(self.predict_prob(x))

    def binary_acc(self, x, y):
        y_true = np.array(y)
        y_pred = self.predict(x).detach().numpy()
        nhits = (y_pred == y_true).sum()
        return nhits / y_true.shape[0]

    def auc(self, x, y):
        scores = self.predict_prob(x)
        return compute_auc(y, scores.detach().numpy())

    def scaler_fit(self, x):
        self.scaler.fit(x)

    def scaler_transform(self, x):
        return self.scaler.transform(x)

    def save_weights(self, f):
        np.save(f, self.alpha_params.detach().numpy())

    def fit(self, x, y, tst_x=None, tst_y=None, nepochs=200, batch_size=256,
        lr=0.001, workers=1, balanced=True, verb=True, scale=False,
        early_stopping=False, patience=10):
        if scale:
            self.scaler_fit(x)
            x = self.scaler_transform(x)
            tst_x = tst_x if tst_x is None else self.scaler_transform(tst_x)
        if balanced:
            n1 = int(sum(y))
            n0 = len(y) - n1
            if n0 < n1:
                p = int(np.floor(n1 / n0))
                X = np.concatenate((x[y == 0].repeat(p, 0), x[y == 1]), 0)
                Y = np.concatenate((y[y == 0].repeat(p, 0), y[y == 1]), 0)
            else:
                p = int(np.floor(n0 / n1))
                X = np.concatenate((x[y == 1].repeat(p, 0), x[y == 0]), 0)
                Y = np.concatenate((y[y == 1].repeat(p, 0), y[y == 0]), 0)
        else:
            X = x
            Y = y
        loader = DataLoader(Subset(torch.tensor(X).float(), torch.Tensor(Y)
            ), batch_size=batch_size, shuffle=True, num_workers=workers)
        criterion = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        best_auc = self.auc(x, y)
        pat = 0
        for epoch in range(nepochs):
            if verb:
                criterion(self(torch.Tensor(x)), torch.Tensor(y)).detach(
                    ).numpy().round(3)
                self.auc(x, y).round(3)
                self.binary_acc(x, y).round(3)
                np.NAN if tst_x is None else self.auc(tst_x, tst_y).round(3)
                None
            if early_stopping:
                cur_auc = self.auc(x, y)
                if cur_auc < best_auc:
                    if pat < patience:
                        pat += 1
                    else:
                        if verb:
                            None
                        return
                else:
                    best_auc = cur_auc
                    pat = 0
            for batch in loader:
                _x, _y = batch['ims'], batch['labels']
                opt.zero_grad()
                pred = self(_x)
                loss = criterion(pred, _y)
                loss.backward()
                opt.step()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
