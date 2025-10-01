import torch
import numpy as np
from torch import nn
import torch as tc
from sklearn.metrics import *
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler


class myDataset(torch.utils.data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MILLR(tc.nn.Module):

    def __init__(self, input_dim, flight_length, device, aggregation=
        'maxpool', output_resize=False):
        super(MILLR, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.flight_length = flight_length
        self.D = input_dim
        self.device = device
        self.task = 'binary'
        self.threshold = 0.5
        self.agg = aggregation
        self.output_resize = output_resize

    def forward(self, x, train=True):
        _N, _, _ = x.size()
        self.pi = self.sigmoid(self.fc(x)).squeeze()
        self.proba_time = self.pi
        if self.agg == 'mean':
            p = tc.mean(self.pi, axis=-1)
        elif self.agg == 'maxpool':
            p = tc.max(self.pi, dim=-1)[0]
        p = p.view(-1, 1)
        return p

    def get_feature_importance(self, columns, n_top=5):
        coeffs = self.fc.weight.flatten().detach().numpy()
        sorted_feat_idx = np.argsort(coeffs)[::-1]
        sorted_columns = columns[sorted_feat_idx[:n_top]]
        top_values = coeffs[sorted_feat_idx[:n_top]]
        return sorted_columns, top_values

    def cross_time_steps_loss(self, Pi):
        diff = (Pi[:, :-1] - Pi[:, 1:]) ** 2
        return tc.mean(tc.mean(diff, axis=-1))

    def train_LR(self, X_train, y_train, X_val, y_val, batch_size,
        print_every_epochs=5, l2=0.001, learning_rate=0.001,
        use_stratified_batch_size=True, verbose=1, num_epochs=100,
        optimizer='adam', momentum=0.99):
        self.train()
        if 'cuda' in self.device:
            self
        else:
            self.cpu()
        self.batch_size = batch_size
        criterion = nn.BCELoss()
        if optimizer == 'adam':
            optimizer = tc.optim.Adam(self.parameters(), lr=learning_rate,
                weight_decay=l2)
        else:
            optimizer = tc.optim.SGD(self.parameters(), momentum=momentum,
                lr=learning_rate, weight_decay=l2)
        hist = np.zeros(num_epochs)
        val_hist = np.zeros(num_epochs)
        b_acc = np.zeros(num_epochs)
        val_b_acc = np.zeros(num_epochs)
        f1 = np.zeros(num_epochs)
        val_f1 = np.zeros(num_epochs)
        if not tc.is_tensor(X_train):
            X_train = tc.Tensor(X_train)
        if not tc.is_tensor(y_train):
            y_train = tc.Tensor(y_train.flatten())
        if X_val is not None:
            if not tc.is_tensor(X_val):
                X_val = tc.Tensor(X_val)
            if not tc.is_tensor(y_val):
                y_val = tc.Tensor(y_val)
            data_val = myDataset(X_val, y_val)
        data_train = myDataset(X_train, y_train)
        if use_stratified_batch_size is False:
            None
            dataloader_train = DataLoader(data_train, batch_size=self.
                batch_size, shuffle=True)
        else:
            None
            weights = []
            for label in tc.unique(y_train):
                count = len(tc.where(y_train == label)[0])
                weights.append(1 / count)
            weights = tc.tensor(weights)
            samples_weights = weights[y_train.type(tc.LongTensor)]
            sampler = WeightedRandomSampler(samples_weights, len(
                samples_weights), replacement=True)
            dataloader_train = DataLoader(data_train, batch_size=self.
                batch_size, sampler=sampler)
        if X_val is not None:
            dataloader_val = DataLoader(data_val, batch_size=self.
                batch_size, shuffle=False)
        try:
            for epoch in tqdm(range(num_epochs)):
                batch_acc = []
                batch_val_acc = []
                batch_f1 = []
                batch_val_f1 = []
                for iteration, (batch_x, batch_y) in enumerate(dataloader_train
                    ):
                    batch_x, batch_y = batch_x, batch_y
                    if epoch == 0 and iteration == 0:
                        for c in tc.unique(y_train):
                            None
                    outputs = self.forward(batch_x)
                    g_loss = self.cross_time_steps_loss(self.pi)
                    loss = criterion(outputs.flatten(), batch_y.view(-1).
                        flatten()) + g_loss
                    hist[epoch] = loss.item()
                    if 'cuda' in self.device:
                        temp_outpouts = (outputs.cpu().detach().numpy() >
                            self.threshold).astype(int)
                        y_batch = batch_y.view(-1).cpu().detach().numpy()
                        b_acc[epoch] = balanced_accuracy_score(y_batch,
                            temp_outpouts)
                    else:
                        temp_outpouts = (outputs.detach().numpy() > self.
                            threshold).astype(int)
                        y_batch = batch_y.view(-1).detach().numpy()
                        b_acc[epoch] = balanced_accuracy_score(y_batch,
                            temp_outpouts)
                    batch_acc.append(b_acc[epoch])
                    batch_f1.append(f1_score(y_batch, temp_outpouts,
                        average='binary'))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if X_val is not None:
                        with tc.no_grad():
                            mini_loss = []
                            for batch_X_val, batch_y_val in dataloader_val:
                                batch_X_val, batch_y_val = (batch_X_val,
                                    batch_y_val)
                                self.valYhat = self.forward(batch_X_val)
                                g_loss_val = self.cross_time_steps_loss(self.pi
                                    )
                                val_loss = criterion(self.valYhat,
                                    batch_y_val.flatten()) + g_loss_val
                                mini_loss.append(val_loss.item())
                                if self.task == 'binary':
                                    if 'cuda' in self.device:
                                        temp_out_y = (self.valYhat.cpu().detach
                                            ().numpy() > self.threshold).astype(int
                                            )
                                        y_val_batch = batch_y_val.view(-1).cpu(
                                            ).detach().numpy()
                                        val_b_acc[epoch] = balanced_accuracy_score(
                                            y_val_batch, temp_out_y)
                                    else:
                                        temp_out_y = (self.valYhat.detach().
                                            numpy() > self.threshold).astype(int)
                                        y_val_batch = batch_y_val.view(-1).detach(
                                            ).numpy()
                                        val_b_acc[epoch] = balanced_accuracy_score(
                                            y_val_batch, temp_out_y)
                                    batch_val_acc.append(val_b_acc[epoch])
                                    batch_val_f1.append(f1_score(
                                        y_val_batch, temp_out_y, average=
                                        'binary'))
                            val_hist[epoch] = np.mean(mini_loss)
                    if verbose == 1:
                        if self.task == 'binary':
                            if epoch % 10 == 0:
                                None
                                None
                                None
                                if X_val is not None:
                                    None
                                    None
                                    None
                        elif epoch % print_every_epochs == 0:
                            None
                if self.task == 'binary':
                    b_acc[epoch] = np.mean(batch_acc)
                    val_b_acc[epoch] = np.mean(batch_val_acc)
                    f1[epoch] = np.mean(batch_f1)
                    val_f1[epoch] = np.mean(batch_val_f1)
        except KeyboardInterrupt:
            self.cpu()
            self.device = 'cpu'
            self.eval()
            self.x_train = X_train
            self.x_test = X_val
            self.hist = hist
            self.val_hist = val_hist
        except:
            raise
        self.cpu()
        self.device = 'cpu'
        self.eval()
        self.x_train = X_train
        self.x_test = X_val
        self.hist = hist
        self.val_hist = val_hist

    def fit(self, **kw):
        self.train_LR(**kw)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'flight_length': 4, 'device': 0}]
