import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict


class MLP(nn.Module):

    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(self.
            input_size, 256)), ('relu1', nn.ReLU()), ('linear2', nn.Linear(
            256, 128)), ('relu2', nn.ReLU()), ('linear3', nn.Linear(128,
            self.output_size))]))

    def forward(self, text):
        out = self.mlp(text)
        return out

    def construct_sparse_tensor(self, coo):
        """import torch
        import numpy as np
        from scipy.sparse import coo_matrix"""
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    def train_mlp(self, x_train: 'torch.Tensor', x_val: 'torch.Tensor',
        x_test: 'torch.Tensor', y_train: 'torch.Tensor', y_val:
        'torch.Tensor', y_test: 'torch.Tensor', hparams: 'dict'):
        lr = hparams['lr']
        epochs = hparams['epochs']
        batch_size = hparams['batch_size']
        patience = hparams['patience']
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_acc = np.NINF
        trace_train = []
        trace_val = []
        for epoch in range(epochs):
            running_train_loss = 0.0
            running_train_acc = 0.0
            None
            for i in tqdm(range(0, x_train.shape[0], batch_size)):
                batch = x_train[i:i + batch_size]
                label = y_train[i:i + batch_size]
                output = self.forward(batch)
                loss = criterion(output, label)
                predictions = output.argmax(axis=1)
                running_train_acc += (predictions == label).sum()
                optimizer.zero_grad()
                loss.backward()
                running_train_loss += loss.item()
                optimizer.step()
            running_val_loss = 0.0
            running_val_acc = 0.0
            for i in tqdm(range(0, x_val.shape[0], batch_size)):
                batch = x_val[i:i + batch_size]
                label = y_val[i:i + batch_size]
                output = self.forward(batch)
                predictions = output.argmax(axis=1)
                running_val_acc += (predictions == label).sum()
                loss = criterion(output, label)
                running_val_loss += loss.item()
            None
            trace_train.append(running_train_loss)
            trace_val.append(running_val_loss)
            if running_val_acc > best_acc:
                best_acc = running_val_acc
                best_epoch = epoch
                best_state = {key: value.cpu() for key, value in self.
                    state_dict().items()}
            elif epoch >= best_epoch + patience:
                break
        self.load_state_dict(best_state)
        torch.save(best_state, 'model.pt')
        predictions = self.forward(x_test).argmax(axis=1)
        (predictions == y_test).sum()
        None
        None
        return trace_train, trace_val


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
