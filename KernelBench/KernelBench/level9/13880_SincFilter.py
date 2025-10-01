import torch
import numpy as np
import torch.utils.data
import torch.nn as torch_nn


class SincFilter(torch_nn.Module):
    """ SincFilter
        Given the cut-off-frequency, produce the low-pass and high-pass
        windowed-sinc-filters.
        If input cut-off-frequency is (batchsize=1, signal_length, 1),
        output filter coef is (batchsize=1, signal_length, filter_order).
        For each time step in [1, signal_length), we calculate one
        filter for low-pass sinc filter and another for high-pass filter.
        
        Example:
        import scipy
        import scipy.signal
        import numpy as np
        
        filter_order = 31
        cut_f = 0.2
        sinc_layer = SincFilter(filter_order)
        lp_coef, hp_coef = sinc_layer(torch.ones(1, 10, 1) * cut_f)
        
        w, h1 = scipy.signal.freqz(lp_coef[0, 0, :].numpy(), [1])
        w, h2 = scipy.signal.freqz(hp_coef[0, 0, :].numpy(), [1])
        plt.plot(w, 20*np.log10(np.abs(h1)))
        plt.plot(w, 20*np.log10(np.abs(h2)))
        plt.plot([cut_f * np.pi, cut_f * np.pi], [-100, 0])
    """

    def __init__(self, filter_order):
        super(SincFilter, self).__init__()
        self.half_k = (filter_order - 1) // 2
        self.order = self.half_k * 2 + 1

    def hamming_w(self, n_index):
        """ prepare hamming window for each time step
        n_index (batchsize=1, signal_length, filter_order)
            For each time step, n_index will be [-(M-1)/2, ... 0, (M-1)/2]
            n_index[0, 0, :] = [-(M-1)/2, ... 0, (M-1)/2]
            n_index[0, 1, :] = [-(M-1)/2, ... 0, (M-1)/2]
            ...
        output  (batchsize=1, signal_length, filter_order)
            output[0, 0, :] = hamming_window
            output[0, 1, :] = hamming_window
            ...
        """
        return 0.54 + 0.46 * torch.cos(2 * np.pi * n_index / self.order)

    def sinc(self, x):
        """ Normalized sinc-filter sin( pi * x) / pi * x
        https://en.wikipedia.org/wiki/Sinc_function
        
        Assume x (batchsize, signal_length, filter_order) and 
        x[0, 0, :] = [-half_order, - half_order+1, ... 0, ..., half_order]
        x[:, :, self.half_order] -> time index = 0, sinc(0)=1
        """
        y = torch.zeros_like(x)
        y[:, :, 0:self.half_k] = torch.sin(np.pi * x[:, :, 0:self.half_k]) / (
            np.pi * x[:, :, 0:self.half_k])
        y[:, :, self.half_k + 1:] = torch.sin(np.pi * x[:, :, self.half_k + 1:]
            ) / (np.pi * x[:, :, self.half_k + 1:])
        y[:, :, self.half_k] = 1
        return y

    def forward(self, cut_f):
        """ lp_coef, hp_coef = forward(self, cut_f)
        cut-off frequency cut_f (batchsize=1, length, dim = 1)
    
        lp_coef: low-pass filter coefs  (batchsize, length, filter_order)
        hp_coef: high-pass filter coefs (batchsize, length, filter_order)
        """
        with torch.no_grad():
            lp_coef = torch.arange(-self.half_k, self.half_k + 1, device=
                cut_f.device)
            lp_coef = lp_coef.repeat(cut_f.shape[0], cut_f.shape[1], 1)
            hp_coef = torch.arange(-self.half_k, self.half_k + 1, device=
                cut_f.device)
            hp_coef = hp_coef.repeat(cut_f.shape[0], cut_f.shape[1], 1)
            tmp_one = torch.pow(-1, hp_coef)
        lp_coef = cut_f * self.sinc(cut_f * lp_coef) * self.hamming_w(lp_coef)
        hp_coef = (self.sinc(hp_coef) - cut_f * self.sinc(cut_f * hp_coef)
            ) * self.hamming_w(hp_coef)
        lp_coef_norm = torch.sum(lp_coef, axis=2).unsqueeze(-1)
        hp_coef_norm = torch.sum(hp_coef * tmp_one, axis=2).unsqueeze(-1)
        lp_coef = lp_coef / lp_coef_norm
        hp_coef = hp_coef / hp_coef_norm
        return lp_coef, hp_coef


def get_inputs():
    return [torch.rand([4, 4, 3])]


def get_init_inputs():
    return [[], {'filter_order': 4}]
