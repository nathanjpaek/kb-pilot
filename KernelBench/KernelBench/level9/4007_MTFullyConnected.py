import time
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.nn import functional as F


class Base(nn.Module):
    """ This class is the base structure for all of classification/regression DNN models.
    Mainly, it provides the general methods for training, evaluating model and predcting the given data.
    """

    def fit(self, train_loader, valid_loader, out, epochs=100, lr=0.0001):
        """Training the DNN model, similar to the scikit-learn or Keras style.
        In the end, the optimal value of parameters will also be persisted on the hard drive.

        Arguments:
            train_loader (DataLoader): Data loader for training set,
                including m X n target FloatTensor and m X l label FloatTensor
                (m is the No. of sample, n is the No. of features, l is the No. of classes or tasks)
            valid_loader (DataLoader): Data loader for validation set.
                The data structure is as same as loader_train.
            out (str): the file path for the model file (suffix with '.pkg')
                and log file (suffix with '.log').
            epochs(int, optional): The maximum of training epochs (default: 100)
            lr (float, optional): learning rate (default: 1e-4)
        """
        if 'optim' in self.__dict__:
            optimizer = self.optim
        else:
            optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = np.inf
        last_save = 0
        if not os.path.exists(out):
            try:
                os.makedirs(out)
            except PermissionError:
                None
        log = open(file=out + '.log', mode='w+')
        for epoch in range(epochs):
            time.time()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (1 - 1 / epochs) ** (epoch * 10)
            for i, (Xb, yb) in enumerate(train_loader):
                Xb, yb = Xb, yb
                optimizer.zero_grad()
                y_ = self(Xb, istrain=True)
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]
                wb = torch.Tensor(yb.size())
                wb[yb == 3.99] = 0.1
                wb[yb != 3.99] = 1
                loss = self.criterion(y_ * wb, yb * wb)
                loss.backward()
                optimizer.step()
            loss_valid = self.evaluate(valid_loader)
            None
            if loss_valid < best_loss:
                torch.save(self.state_dict(), out + '.pkg')
                None
                best_loss = loss_valid
                last_save = epoch
            else:
                None
                if epoch - last_save > 100:
                    break
        log.close()
        self.load_state_dict(torch.load(out + '.pkg'))

    def evaluate(self, loader):
        """Evaluating the performance of the DNN model.

        Arguments:
            loader (torch.util.data.DataLoader): data loader for test set,
                including m X n target FloatTensor and l X n label FloatTensor
                (m is the No. of sample, n is the No. of features, l is the No. of classes or tasks)

        Return:
            loss (float): the average loss value based on the calculation of loss function with given test set.
        """
        loss = 0
        for Xb, yb in loader:
            Xb, yb = Xb, yb
            y_ = self.forward(Xb)
            ix = yb == yb
            yb, y_ = yb[ix], y_[ix]
            wb = torch.Tensor(yb.size())
            wb[yb == 3.99] = 0.1
            wb[yb != 3.99] = 1
            loss += self.criterion(y_ * wb, yb * wb).item()
        loss = loss / len(loader)
        return loss

    def predict(self, loader):
        """Predicting the probability of each sample in the given dataset.

        Arguments:
            loader (torch.util.data.DataLoader): data loader for test set,
                only including m X n target FloatTensor
                (m is the No. of sample, n is the No. of features)

        Return:
            score (ndarray): probability of each sample in the given dataset,
                it is a m X l FloatTensor (m is the No. of sample, l is the No. of classes or tasks.)
        """
        score = []
        for Xb, yb in loader:
            Xb = Xb
            y_ = self.forward(Xb)
            score.append(y_.detach().cpu())
        score = torch.cat(score, dim=0).numpy()
        return score


class MTFullyConnected(Base):
    """Multi-task DNN classification/regression model. It contains four fully connected layers
    between which are dropout layer for robustness.

    Arguments:
        n_dim (int): the No. of columns (features) for input tensor
        n_task (int): the No. of columns (tasks) for output tensor.
        is_reg (bool, optional): Regression model (True) or Classification model (False)
    """

    def __init__(self, n_dim, n_task, is_reg=False):
        super(MTFullyConnected, self).__init__()
        self.n_task = n_task
        self.dropout = nn.Dropout(0.25)
        self.fc0 = nn.Linear(n_dim, 4000)
        self.fc1 = nn.Linear(4000, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.output = nn.Linear(1000, n_task)
        self.is_reg = is_reg
        if is_reg:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCELoss()
            self.activation = nn.Sigmoid()
        self

    def forward(self, X, istrain=False):
        """Invoke the class directly as a function

        Arguments:
            X (FloatTensor): m X n FloatTensor, m is the No. of samples, n is the No. of features.
            istrain (bool, optional): is it invoked during training process (True)
                or just for prediction (False)

        Return:
            y (FloatTensor): m X l FloatTensor, m is the No. of samples, n is the No. of tasks
        """
        y = F.relu(self.fc0(X))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc1(y))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc2(y))
        if istrain:
            y = self.dropout(y)
        if self.is_reg:
            y = self.output(y)
        else:
            y = self.activation(self.output(y))
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_dim': 4, 'n_task': 4}]
