import torch
import torch.nn as nn
import torch.autograd


class GammaScaleMap(nn.Module):
    """
        Compute Gamma Scale on a 4D tensor (The hard way). This acts as a standard PyTorch layer. 
        
        Gamma Scale is computed independantly for each batch item at each location x,y
    
        Input should be:
        
        (1) A tensor of size [batch x channels x height x width] 
        (2) A tensor with only positive values. (After a ReLU)
        
        Output is a 3D tensor of size [batch x height x width] 
    """

    def __init__(self, run_relu=False):
        super(GammaScaleMap, self).__init__()
        """
            SMOE Scale must take in values > 0. Optionally, we can run a ReLU to do that.
        """
        if run_relu:
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = None

    def _trigamma(self, x):
        """
            We need this line since recursion is not good for x < 1.0
            Note that we take + torch.reciprocal(x.pow(2)) at the end because:
        
                trigamma(z) = trigamma(z + 1) + 1/z^2
        
        """
        z = x + 1.0
        zz = z.pow(2)
        a = 0.2 - torch.reciprocal(7.0 * zz)
        b = 1.0 - a / zz
        c = 1.0 + b / (3.0 * z)
        d = 1.0 + c / (2.0 * z)
        e = d / z
        e = e + torch.reciprocal(x.pow(2.0))
        return e

    def _k_update(self, k, s):
        nm = torch.log(k) - torch.digamma(k) - s
        dn = torch.reciprocal(k) - self._trigamma(k)
        k2 = k - nm / dn
        return k2

    def _compute_k_est(self, x, i=10, dim=1):
        """
            Calculate s
        """
        s = torch.log(torch.mean(x, dim=dim)) - torch.mean(torch.log(x),
            dim=dim)
        """
            Get estimate of k to within 1.5%
        
            NOTE: K gets smaller as log variance s increases
        """
        s3 = s - 3.0
        rt = torch.sqrt(s3.pow(2) + 24.0 * s)
        nm = 3.0 - s + rt
        dn = 12.0 * s
        k = nm / dn + 1e-07
        """
            Do i Newton-Raphson steps to get closer than 1.5%
            For i=5 gets us within 4 or 5 decimal places
        """
        for _ in range(i):
            k = self._k_update(k, s)
        return k

    def forward(self, x):
        assert torch.is_tensor(x), 'input must be a Torch Tensor'
        assert len(x.size()) > 2, 'input must have at least three dims'
        """
            If we do not have a convenient ReLU to pluck from, we can do it here
        """
        if self.relu is not None:
            x = self.relu(x)
        """
            avoid log(0)
        """
        x = x + 1e-07
        k = self._compute_k_est(x)
        th = torch.reciprocal(k) * torch.mean(x, dim=1)
        return th


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
