from torch.nn import Module
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class MyNetwork(Module):

    def __init__(self, size_input, size_hidden, size_output):
        """Create simple network"""
        super().__init__()
        self.layer_1 = nn.Linear(size_input, size_hidden)
        self.layer_2 = nn.Linear(size_hidden, size_output)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        """Forward through network"""
        out = self.layer_1(X)
        out = self.layer_2(out)
        out = self.softmax(out)
        return out


class VariableDataLoader(object):
    """Load data from variable length inputs

        Attributes
        ----------
        lengths : dict()
            Dictionary of input-length -> input samples

        index : boolean, default=False
            If True, also returns original index

        batch_size : int, default=1
            Size of each batch to output

        shuffle : boolean, default=True
            If True, shuffle the data randomly, each yielded batch contains
            only input items of the same length
        """

    def __init__(self, X, y, index=False, batch_size=1, shuffle=True):
        """Load data from variable length inputs

            Parameters
            ----------
            X : iterable of shape=(n_samples,)
                Input sequences
                Each item in iterable should be a sequence (of variable length)

            y : iterable of shape=(n_samples,)
                Labels corresponding to X

            index : boolean, default=False
                If True, also returns original index

            batch_size : int, default=1
                Size of each batch to output

            shuffle : boolean, default=True
                If True, shuffle the data randomly, each yielded batch contains
                only input items of the same length
            """
        self.lengths = dict()
        for i, (X_, y_) in enumerate(zip(X, y)):
            X_length, y_length, i_length = self.lengths.get(len(X_), (list(
                ), list(), list()))
            X_length.append(X_)
            y_length.append(y_)
            i_length.append(i)
            self.lengths[len(X_)] = X_length, y_length, i_length
        for k, v in self.lengths.items():
            self.lengths[k] = torch.as_tensor(v[0]), torch.as_tensor(v[1]
                ), torch.as_tensor(v[2])
        self.index = index
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reset()
        self.keys = set(self.data.keys())

    def reset(self):
        """Reset the VariableDataLoader"""
        self.done = set()
        self.data = {k: iter(DataLoader(TensorDataset(v[0], v[1], v[2]),
            batch_size=self.batch_size, shuffle=self.shuffle)) for k, v in
            self.lengths.items()}

    def __iter__(self):
        """Returns iterable of VariableDataLoader"""
        self.reset()
        return self

    def __next__(self):
        """Get next item of VariableDataLoader"""
        if self.done == self.keys:
            self.reset()
            raise StopIteration
        if self.shuffle:
            key = random.choice(list(self.keys - self.done))
        else:
            key = sorted(self.keys - self.done)[0]
        try:
            X_, y_, i = next(self.data.get(key))
            if self.index:
                item = X_, y_, i
            else:
                item = X_, y_
        except StopIteration:
            self.done.add(key)
            item = next(self)
        return item


class Module(nn.Module):
    """Extention of nn.Module that adds fit and predict methods
        Can be used for automatic training.

        Attributes
        ----------
        progress : Progress()
            Used to track progress of fit and predict methods
    """

    def __init__(self, *args, **kwargs):
        """Only calls super method nn.Module with given arguments."""
        super().__init__(*args, **kwargs)

    def fit(self, X, y, epochs=10, batch_size=32, learning_rate=0.01,
        criterion=nn.NLLLoss(), optimizer=optim.SGD, variable=False,
        verbose=True, **kwargs):
        """Train the module with given parameters

            Parameters
            ----------
            X : torch.Tensor
                Tensor to train with

            y : torch.Tensor
                Target tensor

            epochs : int, default=10
                Number of epochs to train with

            batch_size : int, default=32
                Default batch size to use for training

            learning_rate : float, default=0.01
                Learning rate to use for optimizer

            criterion : nn.Loss, default=nn.NLLLoss()
                Loss function to use

            optimizer : optim.Optimizer, default=optim.SGD
                Optimizer to use for training

            variable : boolean, default=False
                If True, accept inputs of variable length

            verbose : boolean, default=True
                If True, prints training progress

            Returns
            -------
            result : self
                Returns self
            """
        optimizer = optimizer(params=self.parameters(), lr=learning_rate)
        if variable:
            'cuda' if torch.cuda.is_available() else 'cpu'
            data = VariableDataLoader(X, y, batch_size=batch_size, shuffle=True
                )
        else:
            data = DataLoader(TensorDataset(X, y), batch_size=batch_size,
                shuffle=True)
        for epoch in range(1, epochs + 1):
            try:
                for X_, y_ in tqdm.tqdm(data, desc=
                    '[Epoch {:{width}}/{:{width}}]'.format(epoch, epochs,
                    width=len(str(epochs)))):
                    optimizer.zero_grad()
                    X_ = X_.clone().detach()
                    y_pred = self(X_)
                    loss = criterion(y_pred, y_)
                    loss.backward()
                    optimizer.step()
            except KeyboardInterrupt:
                None
                break
        return self

    def predict(self, X, batch_size=32, variable=False, verbose=True, **kwargs
        ):
        """Makes prediction based on input data X.
            Default implementation just uses the module forward(X) method,
            often the predict method will be overwritten to fit the specific
            needs of the module.

            Parameters
            ----------
            X : torch.Tensor
                Tensor from which to make prediction

            batch_size : int, default=32
                Batch size in which to predict items in X

            variable : boolean, default=False
                If True, accept inputs of variable length

            verbose : boolean, default=True
                If True, print progress of prediction

            Returns
            -------
            result : torch.Tensor
                Resulting prediction
            """
        with torch.no_grad():
            result = list()
            indices = torch.arange(len(X))
            if variable:
                indices = list()
                data = VariableDataLoader(X, torch.zeros(len(X)), index=
                    True, batch_size=batch_size, shuffle=False)
                for X_, y_, i in tqdm.tqdm(data, desc='Predicting'):
                    result.append(self(X_))
                    indices.append(i)
                indices = torch.cat(indices)
            else:
                for batch in tqdm.tqdm(range(0, X.shape[0], batch_size),
                    desc='Predicting'):
                    X_ = X[batch:batch + batch_size]
                    result.append(self(X_))
            return torch.cat(result)[indices]

    def fit_predict(self, X, y, epochs=10, batch_size=32, learning_rate=
        0.01, criterion=nn.NLLLoss, optimizer=optim.SGD, variable=False,
        verbose=True, **kwargs):
        """Train the module with given parameters

            Parameters
            ----------
            X : torch.Tensor
                Tensor to train with

            y : torch.Tensor
                Target tensor

            epochs : int, default=10
                Number of epochs to train with

            batch_size : int, default=32
                Default batch size to use for training

            learning_rate : float, default=0.01
                Learning rate to use for optimizer

            criterion : nn.Loss, default=nn.NLLLoss
                Loss function to use

            optimizer : optim.Optimizer, default=optim.SGD
                Optimizer to use for training

            variable : boolean, default=False
                If True, accept inputs of variable length

            verbose : boolean, default=True
                If True, prints training progress

            Returns
            -------
            result : torch.Tensor
                Resulting prediction
            """
        return self.fit(X, y, epochs, batch_size, learning_rate, criterion,
            optimizer, variable, verbose, **kwargs).predict(X, batch_size,
            variable, verbose, **kwargs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size_input': 4, 'size_hidden': 4, 'size_output': 4}]
