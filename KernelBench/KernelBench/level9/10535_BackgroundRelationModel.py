import torch
import numpy as np
from torch import nn
from torch.nn.parameter import Parameter


class BackgroundRelationModel(nn.Module):

    def __init__(self, n_bg, n_ml):
        """
        n_bg: number of background tags
        n_ml: number of ml tags
        """
        super().__init__()
        self.config = {'n_bg': n_bg, 'n_ml': n_ml}
        self.W = Parameter(torch.randn(n_bg, n_ml))
        self.b = Parameter(torch.randn(n_ml) / 2)
        nn.init.xavier_uniform_(self.W)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self

    def forward(self, alpha_bg):
        """
        The inputs to the forward function should be:
            the alpha matrix (np array) with columns corresponding to the background tags
            it should have the shape (batch_size, n_bg)
        """
        alpha_bg_tensor = torch.tensor(alpha_bg)
        return alpha_bg_tensor @ self.W + self.b

    def fit(self, alpha_bg, alpha_ml, lr=0.001, epochs=10):
        """
        alpha_bg: the alpha matrix (np array) with columns corresponding to the background tags
        alpha_ml: the alpha matrix (np array) with columns corresponding to the ML tags
        lr: learning rate
        epochs: number of epochs to train
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        ys = torch.tensor(alpha_ml)
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_hats = self.forward(alpha_bg)
            loss = loss_fn(y_hats, ys)
            loss.backward()
            optimizer.step()
            None

    def auto_fit(self, alpha_bg, alpha_ml, val_alpha_bg, val_alpha_ml, lr=
        0.001, reg=0.01, patience=10):
        """
        alpha_bg: the alpha matrix (np array) with columns corresponding to the background tags
        alpha_ml: the alpha matrix (np array) with columns corresponding to the ML tags
        val_*: same data but from the validation set
        lr: learning rate
        reg: weight decay coefficient (similar to L2 penalty)
        patience: number of epochs to continue evaluating even after loss not decreasing
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=reg
            )
        loss_fn = nn.MSELoss()
        epoch = 0
        frustration = 0
        best_loss = np.inf
        ys = torch.tensor(alpha_ml)
        val_ys = torch.tensor(val_alpha_ml)
        while True:
            optimizer.zero_grad()
            y_hats = self.forward(alpha_bg)
            loss = loss_fn(y_hats, ys)
            loss.backward()
            optimizer.step()
            val_y_hats = self.forward(val_alpha_bg)
            val_loss = loss_fn(val_y_hats, val_ys)
            None
            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                frustration = 0
            else:
                frustration += 1
            if frustration > patience:
                break
            epoch += 1

    def predict(self, alpha_bg):
        self.eval()
        return self.forward(alpha_bg).cpu().detach().numpy()

    def save(self, model_path):
        model_state = {'state_dict': self.state_dict(), 'config': self.config}
        torch.save(model_state, model_path)

    @classmethod
    def load(cls, model_path):
        model_state = torch.load(str(model_path), map_location=lambda
            storage, loc: storage)
        args = model_state['config']
        model = cls(**args)
        model.load_state_dict(model_state['state_dict'])
        if torch.cuda.is_available():
            model.device = torch.device('cuda')
        else:
            model.device = torch.device('cpu')
        model
        return model


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_bg': 4, 'n_ml': 4}]
