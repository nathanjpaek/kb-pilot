import math
import torch
import torch.nn as nn
import torch.utils.data


class Loss(nn.Module):

    def __init__(self, type_in='pred_intervals', alpha=0.1, loss_type=
        'qd_soft', censor_R=False, soften=100.0, lambda_in=10.0, sigma_in=
        0.5, use_cuda=True):
        super().__init__()
        self.alpha = alpha
        self.lambda_in = lambda_in
        self.soften = soften
        self.loss_type = loss_type
        self.type_in = type_in
        self.censor_R = censor_R
        self.sigma_in = sigma_in
        if use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def forward(self, y_pred, y_true):
        if self.type_in == 'pred_intervals':
            metric = []
            metric_name = []
            y_U = y_pred[:, 0]
            y_L = y_pred[:, 1]
            y_T = y_true[:, 0]
            N_ = y_T.shape[0]
            alpha_ = self.alpha
            lambda_ = self.lambda_in
            torch.mean(y_pred, dim=1)
            MPIW = torch.mean(y_U - y_L)
            gamma_U = torch.sigmoid((y_U - y_T) * self.soften)
            gamma_L = torch.sigmoid((y_T - y_L) * self.soften)
            gamma_ = torch.mul(gamma_U, gamma_L)
            torch.ones_like(gamma_)
            zeros = torch.zeros_like(y_U)
            gamma_U_hard = torch.max(zeros, torch.sign(y_U - y_T))
            gamma_L_hard = torch.max(zeros, torch.sign(y_T - y_L))
            gamma_hard = torch.mul(gamma_U_hard, gamma_L_hard)
            qd_lhs_hard = torch.div(torch.mean(torch.abs(y_U - y_L) *
                gamma_hard), torch.mean(gamma_hard) + 0.001)
            torch.div(torch.mean(torch.abs(y_U - y_L) * gamma_), torch.mean
                (gamma_) + 0.001)
            PICP_soft = torch.mean(gamma_)
            PICP_hard = torch.mean(gamma_hard)
            zero = torch.tensor(0.0)
            qd_rhs_soft = lambda_ * math.sqrt(N_) * torch.pow(torch.max(
                zero, 1.0 - alpha_ - PICP_soft), 2)
            qd_rhs_hard = lambda_ * math.sqrt(N_) * torch.pow(torch.max(
                zero, 1.0 - alpha_ - PICP_hard), 2)
            qd_loss_soft = qd_lhs_hard + qd_rhs_soft
            qd_loss_hard = qd_lhs_hard + qd_rhs_hard
            y_mean = y_U
            y_var_limited = torch.min(y_L, torch.tensor(10.0))
            y_var = torch.max(torch.log(1.0 + torch.exp(y_var_limited)),
                torch.tensor(1e-05))
            self.y_mean = y_mean
            self.y_var = y_var
            gauss_loss = torch.log(y_var) / 2.0 + torch.div(torch.pow(y_T -
                y_mean, 2), 2.0 * y_var)
            gauss_loss = torch.mean(gauss_loss)
            if self.loss_type == 'qd_soft':
                loss = qd_loss_soft
            elif self.loss_type == 'qd_hard':
                loss = qd_loss_hard
            elif self.loss_type == 'gauss_like':
                loss = gauss_loss
            elif self.loss_type == 'picp':
                loss = PICP_hard
            elif self.loss_type == 'mse':
                loss = torch.mean(torch.pow(y_U - y_T, 2))
            torch.mean(gamma_U_hard)
            torch.mean(gamma_L_hard)
            PICP = torch.mean(gamma_hard)
            metric.append(PICP)
            metric_name.append('PICP')
            metric.append(MPIW)
            metric_name.append('MPIW')
        return loss, PICP, MPIW


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
